"""
CRATE Interactive White-Box Inspector.

Standalone analysis script that loads a CRATE/SelfFlowCRATE checkpoint,
runs forward passes with hooks to capture all interpretable internals,
and generates a self-contained interactive HTML explorer.

Usage:
    python -m scripts.crate_inspect --source base --model-tag d20 \
        --text "The capital of France is Paris, which is known for"

    python -m scripts.crate_inspect --checkpoint-dir /path/to/ckpt --step 5000 \
        --text "Hello world" --output my_report.html
"""

import argparse
import base64
import io
import json
import math
import os
import sys
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from nanochat.checkpoint_manager import load_model, load_model_from_dir
from nanochat.tokenizer import get_tokenizer
from nanochat.common import autodetect_device_type
from nanochat.crate import norm


# ---------------------------------------------------------------------------
# Data collection via forward hooks
# ---------------------------------------------------------------------------

class CRATEInspector:
    """Collects interpretable internals from a CRATE forward pass."""

    def __init__(self, model):
        self.model = model
        self.backbone = getattr(model, "backbone", model)
        self.config = self.backbone.config
        self._hooks = []
        self._data = {}

    def _get_blocks(self):
        return list(self.backbone.transformer.h)

    def _register_hooks(self):
        blocks = self._get_blocks()
        n_layer = len(blocks)
        n_head = self.config.n_head
        head_dim = self.config.n_embd // n_head

        self._data = {
            "hidden_states": [None] * (n_layer + 1),
            "attention_weights": [None] * n_layer,
            "odl_activations": [None] * n_layer,
            "odl_pre_threshold": [None] * n_layer,
        }

        # Hook: capture input to first block (post-embedding hidden state)
        def capture_embedding_hook(layer_idx):
            def hook(module, args):
                x = args[0]
                if self._data["hidden_states"][0] is None:
                    self._data["hidden_states"][0] = x.detach().float().cpu()
            return hook

        h0 = self._hooks.append(blocks[0].register_forward_pre_hook(capture_embedding_hook(0)))

        for layer_idx, block in enumerate(blocks):
            # Hook: capture output hidden state after each block
            def capture_hidden_hook(idx):
                def hook(module, args, output):
                    self._data["hidden_states"][idx + 1] = output.detach().float().cpu()
                return hook
            self._hooks.append(block.register_forward_hook(capture_hidden_hook(layer_idx)))

            # Hook: capture MSSA attention weights by recomputing from tied projection
            def capture_attention_hook(idx, n_h, h_d, scale):
                def hook(module, args, output):
                    x = args[0]
                    B, T, C = x.size()
                    w = module.qkv(x).view(B, T, n_h, h_d)
                    cos_val = self.backbone.cos[:, :T]
                    sin_val = self.backbone.sin[:, :T]
                    from nanochat.crate import apply_rotary_emb
                    w = apply_rotary_emb(w, cos_val, sin_val)
                    w = norm(w)
                    w = w.transpose(1, 2)  # [B, H, T, D]
                    dots = torch.matmul(w, w.transpose(-1, -2)) * scale
                    causal_mask = torch.triu(
                        torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
                    )
                    dots = dots.masked_fill(causal_mask, float("-inf"))
                    attn = F.softmax(dots, dim=-1)
                    self._data["attention_weights"][idx] = attn[0].detach().float().cpu()
                return hook

            self._hooks.append(
                block.mssa.register_forward_hook(
                    capture_attention_hook(layer_idx, n_head, head_dim, head_dim ** -0.5)
                )
            )

            # Hook: capture ODL sparse activations (post-threshold)
            sparse_module = block.ista if block.sparse_block_type == "ista" else block.odl
            def capture_odl_hook(idx, smod):
                def hook(module, args, output):
                    x = args[0]
                    h = module.D_enc(x) if hasattr(module, "D_enc") else F.linear(x, module.weight, bias=None)
                    self._data["odl_pre_threshold"][idx] = h[0].detach().float().cpu()
                    if hasattr(module, "D_enc"):
                        if module.use_relu:
                            activated = F.relu(h - module.threshold)
                        else:
                            from nanochat.crate import soft_threshold
                            activated = soft_threshold(h, module.step_size * module.lambd)
                    else:
                        activated = output
                    self._data["odl_activations"][idx] = activated[0].detach().float().cpu()
                return hook

            self._hooks.append(sparse_module.register_forward_hook(capture_odl_hook(layer_idx, sparse_module)))

    def _remove_hooks(self):
        for h in self._hooks:
            if h is not None:
                h.remove()
        self._hooks.clear()

    @torch.no_grad()
    def inspect(self, token_ids, device):
        """Run a forward pass and collect all internals."""
        self._register_hooks()
        try:
            x = torch.tensor([token_ids], dtype=torch.long, device=device)
            self.model(x)
        finally:
            self._remove_hooks()
        return self._data

    def extract_thresholds(self):
        """Extract learned ODL threshold parameters."""
        thresholds = []
        for block in self._get_blocks():
            smod = block.ista if block.sparse_block_type == "ista" else block.odl
            if hasattr(smod, "threshold"):
                thresholds.append(smod.threshold.detach().float().cpu().numpy())
            elif hasattr(smod, "lambd"):
                thresholds.append(np.full(smod.dim, smod.step_size * smod.lambd))
            else:
                thresholds.append(np.zeros(self.config.n_embd))
        return thresholds

    def extract_layer_scalars(self):
        """Extract per-layer residual and x0 lambda scalars."""
        return {
            "resid_lambdas": self.backbone.resid_lambdas.detach().float().cpu().numpy().tolist(),
            "x0_lambdas": self.backbone.x0_lambdas.detach().float().cpu().numpy().tolist(),
        }


# ---------------------------------------------------------------------------
# Coding rate computation
# ---------------------------------------------------------------------------

def compute_coding_rate(Z, eps=0.5):
    """
    Compute coding rate R(Z) = (1/2) log det(I + (d/(n*eps^2)) Z^T Z).
    Z: [T, d] hidden states for one sequence.
    """
    T, d = Z.shape
    if T == 0 or d == 0:
        return 0.0
    Z = Z.float()
    cov = (Z.T @ Z) / T  # [d, d]
    scaled = torch.eye(d) + (d / (T * eps ** 2)) * cov
    sign, logabsdet = torch.linalg.slogdet(scaled)
    return 0.5 * logabsdet.item()


# ---------------------------------------------------------------------------
# Data serialization for HTML embedding
# ---------------------------------------------------------------------------

def numpy_to_b64(arr):
    """Encode a numpy array as base64 for compact HTML embedding."""
    buf = io.BytesIO()
    np.save(buf, arr.astype(np.float32))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def prepare_report_data(token_strs, raw_data, thresholds, layer_scalars, config, max_atoms=200):
    """Convert raw hook data into a JSON-serializable report."""
    n_layer = config.n_layer
    n_head = config.n_head
    n_embd = config.n_embd
    T = len(token_strs)

    # Per-layer metrics
    sparsity = []
    coding_rates = []
    attn_maps = []
    odl_top_atoms = []
    odl_sparsity_per_token = []
    threshold_stats = []

    for layer_idx in range(n_layer):
        # Sparsity
        odl_act = raw_data["odl_activations"][layer_idx]  # [T, hidden_dim]
        if odl_act is not None:
            odl_np = odl_act.numpy()
            l0 = float((np.abs(odl_np) < 1e-8).mean())
            sparsity.append(l0)

            # Per-token sparsity
            per_tok = [(np.abs(odl_np[t]) < 1e-8).mean() for t in range(T)]
            odl_sparsity_per_token.append(per_tok)

            # Top-K atoms per token
            layer_top_atoms = []
            for t in range(T):
                magnitudes = np.abs(odl_np[t])
                top_k_idx = np.argsort(magnitudes)[-max_atoms:][::-1]
                top_k_val = magnitudes[top_k_idx]
                mask = top_k_val > 1e-8
                layer_top_atoms.append({
                    "indices": top_k_idx[mask].tolist(),
                    "values": top_k_val[mask].tolist(),
                })
            odl_top_atoms.append(layer_top_atoms)
        else:
            sparsity.append(0.0)
            odl_sparsity_per_token.append([0.0] * T)
            odl_top_atoms.append([{"indices": [], "values": []} for _ in range(T)])

        # Coding rate
        hidden = raw_data["hidden_states"][layer_idx + 1]
        if hidden is not None:
            cr = compute_coding_rate(hidden[0])
            coding_rates.append(cr)
        else:
            coding_rates.append(0.0)

        # Attention (subsample heads if too many)
        attn = raw_data["attention_weights"][layer_idx]
        if attn is not None:
            attn_np = attn.numpy()  # [H, T, T]
            attn_maps.append(attn_np.tolist())
        else:
            attn_maps.append(None)

        # Threshold stats
        if layer_idx < len(thresholds):
            th = thresholds[layer_idx]
            threshold_stats.append({
                "mean": float(np.mean(th)),
                "std": float(np.std(th)),
                "min": float(np.min(th)),
                "max": float(np.max(th)),
                "median": float(np.median(th)),
                "q25": float(np.percentile(th, 25)),
                "q75": float(np.percentile(th, 75)),
                "histogram": np.histogram(th, bins=50)[0].tolist(),
                "bin_edges": np.histogram(th, bins=50)[1].tolist(),
            })
        else:
            threshold_stats.append(None)

    # Initial coding rate (post-embedding)
    h0 = raw_data["hidden_states"][0]
    cr0 = compute_coding_rate(h0[0]) if h0 is not None else 0.0

    # Atom utilization across all layers
    atom_utilization = []
    for layer_idx in range(n_layer):
        odl_act = raw_data["odl_activations"][layer_idx]
        if odl_act is not None:
            fired = (np.abs(odl_act.numpy()) > 1e-8).sum(axis=0)  # [hidden_dim]
            atom_utilization.append({
                "total_atoms": int(fired.shape[0]),
                "active_atoms": int((fired > 0).sum()),
                "mean_activations": float(fired.mean()),
                "top_50": np.argsort(fired)[-50:][::-1].tolist(),
                "top_50_counts": fired[np.argsort(fired)[-50:][::-1]].tolist(),
            })
        else:
            atom_utilization.append(None)

    return {
        "tokens": token_strs,
        "config": {
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embd": n_embd,
            "odl_expansion": getattr(config, "odl_expansion", 1),
            "hidden_dim": n_embd * getattr(config, "odl_expansion", 1),
        },
        "layer_scalars": layer_scalars,
        "sparsity": sparsity,
        "coding_rates": [cr0] + coding_rates,
        "attention_maps": attn_maps,
        "odl_top_atoms": odl_top_atoms,
        "odl_sparsity_per_token": odl_sparsity_per_token,
        "threshold_stats": threshold_stats,
        "atom_utilization": atom_utilization,
    }


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_html(report_data):
    """Generate a self-contained interactive HTML report."""
    data_json = json.dumps(report_data)
    return HTML_TEMPLATE.replace("/*__REPORT_DATA__*/{}", data_json)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CRATE Interactive White-Box Inspector")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", type=str, help="Checkpoint source name (base, mid, semisup, etc.)")
    group.add_argument("--checkpoint-dir", type=str, help="Direct path to checkpoint directory")

    parser.add_argument("--model-tag", type=str, default=None, help="Model tag (e.g. d20)")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (default: latest)")
    parser.add_argument("--text", type=str, nargs="+", required=True, help="Text(s) to analyze")
    parser.add_argument("--output", type=str, default="crate_inspect.html", help="Output HTML path")
    parser.add_argument("--device", type=str, default="", help="Device (cpu/cuda/mps, default: auto)")
    parser.add_argument("--max-atoms", type=int, default=200, help="Max dictionary atoms to display per layer")

    args = parser.parse_args()

    # Device
    device_type = autodetect_device_type() if args.device == "" else args.device
    device = torch.device(device_type)

    # Load model
    print(f"Loading model...")
    if args.source:
        model, tokenizer, meta_data = load_model(
            args.source, device=device, phase="eval",
            model_tag=args.model_tag, step=args.step,
        )
    else:
        model, tokenizer, meta_data = load_model_from_dir(
            args.checkpoint_dir, device=device, phase="eval",
            model_tag=args.model_tag, step=args.step,
        )

    model.eval()
    inspector = CRATEInspector(model)
    config = inspector.config

    print(f"Model: {config.n_layer} layers, d={config.n_embd}, heads={config.n_head}")
    print(f"ODL expansion: {getattr(config, 'odl_expansion', 1)}x -> hidden_dim={config.n_embd * getattr(config, 'odl_expansion', 1)}")

    # Extract static model data
    thresholds = inspector.extract_thresholds()
    layer_scalars = inspector.extract_layer_scalars()

    # Process each text input
    full_text = " ".join(args.text)
    token_ids = tokenizer(full_text, prepend="<|bos|>")
    token_strs = []
    for tid in token_ids:
        try:
            s = tokenizer.decode([tid])
        except Exception:
            s = f"[{tid}]"
        token_strs.append(s)

    print(f"Input: {len(token_ids)} tokens")
    print(f"Tokens: {token_strs}")

    # Run inspection
    print("Running forward pass with hooks...")
    autocast = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
    with autocast:
        raw_data = inspector.inspect(token_ids, device)

    # Prepare report
    print("Computing metrics...")
    report = prepare_report_data(
        token_strs, raw_data, thresholds, layer_scalars, config,
        max_atoms=args.max_atoms,
    )

    # Generate HTML
    print(f"Generating HTML report...")
    html = generate_html(report)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Report saved to {args.output} ({size_mb:.1f} MB)")
    print("Open in a browser to explore.")


# ---------------------------------------------------------------------------
# HTML Template (self-contained with D3.js + Chart.js)
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CRATE White-Box Inspector</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg: #0d1117;
  --bg2: #161b22;
  --bg3: #21262d;
  --border: #30363d;
  --text: #e6edf3;
  --text2: #8b949e;
  --accent: #58a6ff;
  --accent2: #3fb950;
  --accent3: #d29922;
  --accent4: #f85149;
  --font: 'JetBrains Mono', 'Fira Code', 'SF Mono', monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font);
  font-size: 13px;
  line-height: 1.5;
  overflow-x: hidden;
}
.header {
  padding: 20px 24px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 16px;
}
.header h1 {
  font-size: 18px;
  font-weight: 600;
  letter-spacing: -0.5px;
}
.header .meta {
  color: var(--text2);
  font-size: 12px;
}
.token-bar {
  padding: 16px 24px;
  border-bottom: 1px solid var(--border);
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  align-items: center;
  background: var(--bg2);
}
.token-chip {
  padding: 3px 6px;
  border-radius: 4px;
  background: var(--bg3);
  border: 1px solid var(--border);
  cursor: pointer;
  font-size: 12px;
  white-space: pre;
  transition: all 0.15s;
}
.token-chip:hover { border-color: var(--accent); color: var(--accent); }
.token-chip.selected {
  background: var(--accent);
  color: var(--bg);
  border-color: var(--accent);
  font-weight: 600;
}
.main {
  display: grid;
  grid-template-columns: 220px 1fr;
  min-height: calc(100vh - 120px);
}
.sidebar {
  border-right: 1px solid var(--border);
  padding: 16px 0;
  background: var(--bg2);
}
.sidebar .nav-item {
  padding: 8px 20px;
  cursor: pointer;
  font-size: 12px;
  color: var(--text2);
  transition: all 0.15s;
  border-left: 2px solid transparent;
}
.sidebar .nav-item:hover { color: var(--text); background: var(--bg3); }
.sidebar .nav-item.active {
  color: var(--accent);
  border-left-color: var(--accent);
  background: var(--bg3);
}
.sidebar .section-label {
  padding: 4px 20px;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: var(--text2);
  margin-top: 12px;
}
.content {
  padding: 24px;
  overflow-y: auto;
  max-height: calc(100vh - 120px);
}
.panel {
  display: none;
  animation: fadeIn 0.2s ease;
}
.panel.active { display: block; }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
.card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 16px;
}
.card h3 {
  font-size: 13px;
  font-weight: 600;
  margin-bottom: 12px;
  color: var(--accent);
}
.chart-container { position: relative; width: 100%; max-height: 350px; }
.grid-2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
.heatmap-container {
  overflow-x: auto;
  position: relative;
}
.heatmap-canvas { image-rendering: pixelated; }
.layer-selector {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
  margin-bottom: 16px;
}
.layer-btn {
  padding: 4px 10px;
  border-radius: 4px;
  background: var(--bg3);
  border: 1px solid var(--border);
  color: var(--text2);
  cursor: pointer;
  font-size: 11px;
  font-family: var(--font);
  transition: all 0.15s;
}
.layer-btn:hover { border-color: var(--accent); color: var(--accent); }
.layer-btn.active {
  background: var(--accent);
  color: var(--bg);
  border-color: var(--accent);
}
.head-selector {
  display: flex;
  gap: 3px;
  flex-wrap: wrap;
  margin-bottom: 12px;
}
.head-btn {
  padding: 2px 8px;
  border-radius: 3px;
  background: var(--bg3);
  border: 1px solid var(--border);
  color: var(--text2);
  cursor: pointer;
  font-size: 10px;
  font-family: var(--font);
}
.head-btn:hover { border-color: var(--accent2); color: var(--accent2); }
.head-btn.active {
  background: var(--accent2);
  color: var(--bg);
  border-color: var(--accent2);
}
.stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 8px;
  margin-bottom: 16px;
}
.stat-box {
  background: var(--bg3);
  border-radius: 6px;
  padding: 10px;
  text-align: center;
}
.stat-box .value {
  font-size: 18px;
  font-weight: 700;
  color: var(--accent);
}
.stat-box .label {
  font-size: 10px;
  color: var(--text2);
  margin-top: 2px;
}
table.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}
table.data-table th, table.data-table td {
  padding: 6px 10px;
  border-bottom: 1px solid var(--border);
  text-align: left;
}
table.data-table th {
  color: var(--text2);
  font-weight: 500;
  font-size: 11px;
}
.bar {
  height: 14px;
  background: var(--accent);
  border-radius: 2px;
  min-width: 2px;
  transition: width 0.3s;
}
</style>
</head>
<body>

<div class="header">
  <h1>CRATE White-Box Inspector</h1>
  <span class="meta" id="modelMeta"></span>
</div>

<div class="token-bar" id="tokenBar"></div>

<div class="main">
  <div class="sidebar">
    <div class="section-label">Overview</div>
    <div class="nav-item active" data-panel="overview">Sparsity &amp; Coding Rate</div>
    <div class="nav-item" data-panel="thresholds">Learned Thresholds</div>
    <div class="nav-item" data-panel="scalars">Layer Scalars</div>
    <div class="section-label">Per-Layer</div>
    <div class="nav-item" data-panel="attention">Attention Patterns</div>
    <div class="nav-item" data-panel="dictionary">Dictionary Atoms</div>
    <div class="nav-item" data-panel="utilization">Atom Utilization</div>
    <div class="section-label">Per-Token</div>
    <div class="nav-item" data-panel="tokenDetail">Token Detail</div>
  </div>

  <div class="content" id="contentArea">
    <!-- Overview panel -->
    <div class="panel active" id="panel-overview">
      <div class="grid-2">
        <div class="card">
          <h3>ODL Sparsity (L0) Across Layers</h3>
          <div class="chart-container"><canvas id="chartSparsity"></canvas></div>
        </div>
        <div class="card">
          <h3>Coding Rate R(Z) Across Layers</h3>
          <div class="chart-container"><canvas id="chartCodingRate"></canvas></div>
        </div>
      </div>
      <div class="card">
        <h3>Per-Token Sparsity Heatmap</h3>
        <p style="color:var(--text2);font-size:11px;margin-bottom:8px">Rows = layers, columns = tokens. Brighter = more sparse (more forgetting).</p>
        <div class="layer-selector" id="sparsityLayerSel"></div>
        <canvas id="canvasSparsityHeatmap" class="heatmap-canvas" height="200"></canvas>
      </div>
    </div>

    <!-- Thresholds panel -->
    <div class="panel" id="panel-thresholds">
      <div class="card">
        <h3>Learned ODL Threshold Distribution Per Layer</h3>
        <div class="chart-container"><canvas id="chartThresholds"></canvas></div>
      </div>
      <div class="card">
        <h3>Threshold Statistics</h3>
        <table class="data-table" id="thresholdTable">
          <thead><tr><th>Layer</th><th>Mean</th><th>Std</th><th>Min</th><th>Median</th><th>Max</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </div>

    <!-- Scalars panel -->
    <div class="panel" id="panel-scalars">
      <div class="grid-2">
        <div class="card">
          <h3>Residual Lambda (per layer)</h3>
          <div class="chart-container"><canvas id="chartResidLambda"></canvas></div>
        </div>
        <div class="card">
          <h3>X0 Lambda (per layer)</h3>
          <div class="chart-container"><canvas id="chartX0Lambda"></canvas></div>
        </div>
      </div>
    </div>

    <!-- Attention panel -->
    <div class="panel" id="panel-attention">
      <div class="card">
        <h3>Attention Heatmap</h3>
        <div class="layer-selector" id="attnLayerSel"></div>
        <div class="head-selector" id="attnHeadSel"></div>
        <canvas id="canvasAttn" class="heatmap-canvas"></canvas>
      </div>
    </div>

    <!-- Dictionary panel -->
    <div class="panel" id="panel-dictionary">
      <div class="card">
        <h3>Top Active Dictionary Atoms Per Token</h3>
        <div class="layer-selector" id="dictLayerSel"></div>
        <canvas id="canvasDict" class="heatmap-canvas"></canvas>
      </div>
      <div class="card" id="atomDetailCard" style="display:none">
        <h3>Atom Detail</h3>
        <div id="atomDetailContent"></div>
      </div>
    </div>

    <!-- Utilization panel -->
    <div class="panel" id="panel-utilization">
      <div class="card">
        <h3>Atom Utilization</h3>
        <div class="layer-selector" id="utilLayerSel"></div>
        <div class="stat-grid" id="utilStats"></div>
        <div class="chart-container"><canvas id="chartUtilization"></canvas></div>
      </div>
    </div>

    <!-- Token detail panel -->
    <div class="panel" id="panel-tokenDetail">
      <div id="tokenDetailPlaceholder" class="card">
        <h3>Select a Token</h3>
        <p style="color:var(--text2)">Click any token in the bar above to see its detailed analysis across all layers.</p>
      </div>
      <div id="tokenDetailContent" style="display:none">
        <div class="card">
          <h3 id="tokenDetailTitle"></h3>
          <div class="stat-grid" id="tokenStats"></div>
        </div>
        <div class="grid-2">
          <div class="card">
            <h3>Sparsity Across Layers</h3>
            <div class="chart-container"><canvas id="chartTokenSparsity"></canvas></div>
          </div>
          <div class="card">
            <h3>Active Atoms Across Layers</h3>
            <div class="chart-container"><canvas id="chartTokenAtoms"></canvas></div>
          </div>
        </div>
        <div class="card">
          <h3>Top Atoms at Selected Layer</h3>
          <div class="layer-selector" id="tokenAtomLayerSel"></div>
          <div class="chart-container"><canvas id="chartTokenTopAtoms"></canvas></div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
const DATA = /*__REPORT_DATA__*/{};
const TOKENS = DATA.tokens || [];
const N_LAYER = DATA.config?.n_layer || 0;
const N_HEAD = DATA.config?.n_head || 0;

let selectedToken = -1;
let selectedLayer = 0;
let selectedHead = 0;
let charts = {};

// -- Initialization --
function init() {
  document.getElementById('modelMeta').textContent =
    `${N_LAYER} layers | d=${DATA.config?.n_embd} | ${N_HEAD} heads | ODL ${DATA.config?.odl_expansion}x (hidden=${DATA.config?.hidden_dim})`;

  renderTokenBar();
  setupNavigation();
  renderOverview();
  renderThresholds();
  renderScalars();
  buildLayerSelectors();
  renderAttention();
  renderDictionary();
  renderUtilization();
}

// -- Token bar --
function renderTokenBar() {
  const bar = document.getElementById('tokenBar');
  TOKENS.forEach((tok, i) => {
    const chip = document.createElement('span');
    chip.className = 'token-chip';
    chip.textContent = tok;
    chip.title = `Token ${i}: "${tok}"`;
    chip.onclick = () => selectToken(i);
    bar.appendChild(chip);
  });
}

function selectToken(idx) {
  selectedToken = idx;
  document.querySelectorAll('.token-chip').forEach((c, i) => {
    c.classList.toggle('selected', i === idx);
  });
  renderTokenDetail();
  // Switch to token detail panel
  activatePanel('tokenDetail');
}

// -- Navigation --
function setupNavigation() {
  document.querySelectorAll('.nav-item').forEach(item => {
    item.onclick = () => activatePanel(item.dataset.panel);
  });
}

function activatePanel(panelId) {
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  const navItem = document.querySelector(`.nav-item[data-panel="${panelId}"]`);
  if (navItem) navItem.classList.add('active');
  const panel = document.getElementById(`panel-${panelId}`);
  if (panel) panel.classList.add('active');
}

// -- Layer selectors --
function buildLayerSelectors() {
  ['attnLayerSel', 'dictLayerSel', 'utilLayerSel', 'tokenAtomLayerSel'].forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.innerHTML = '';
    for (let i = 0; i < N_LAYER; i++) {
      const btn = document.createElement('button');
      btn.className = 'layer-btn' + (i === 0 ? ' active' : '');
      btn.textContent = `L${i}`;
      btn.onclick = () => {
        el.querySelectorAll('.layer-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        selectedLayer = i;
        if (id === 'attnLayerSel') renderAttention();
        if (id === 'dictLayerSel') renderDictionary();
        if (id === 'utilLayerSel') renderUtilization();
        if (id === 'tokenAtomLayerSel') renderTokenTopAtoms();
      };
      el.appendChild(btn);
    }
  });
  // Head selector
  const headSel = document.getElementById('attnHeadSel');
  if (headSel) {
    headSel.innerHTML = '';
    for (let h = 0; h < N_HEAD; h++) {
      const btn = document.createElement('button');
      btn.className = 'head-btn' + (h === 0 ? ' active' : '');
      btn.textContent = `H${h}`;
      btn.onclick = () => {
        headSel.querySelectorAll('.head-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        selectedHead = h;
        renderAttention();
      };
      headSel.appendChild(btn);
    }
  }
}

// -- Chart helpers --
function makeLineChart(canvasId, labels, datasets, yLabel) {
  if (charts[canvasId]) charts[canvasId].destroy();
  const ctx = document.getElementById(canvasId)?.getContext('2d');
  if (!ctx) return;
  charts[canvasId] = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: datasets.length > 1, labels: { color: '#8b949e', font: { family: 'monospace', size: 11 } } } },
      scales: {
        x: { ticks: { color: '#8b949e', font: { size: 10 } }, grid: { color: '#21262d' } },
        y: { title: { display: !!yLabel, text: yLabel, color: '#8b949e' }, ticks: { color: '#8b949e', font: { size: 10 } }, grid: { color: '#21262d' } },
      },
    },
  });
}

function makeBarChart(canvasId, labels, values, color, yLabel) {
  if (charts[canvasId]) charts[canvasId].destroy();
  const ctx = document.getElementById(canvasId)?.getContext('2d');
  if (!ctx) return;
  charts[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ data: values, backgroundColor: color || '#58a6ff', borderRadius: 2 }] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#8b949e', font: { size: 9 }, maxRotation: 45 }, grid: { display: false } },
        y: { title: { display: !!yLabel, text: yLabel, color: '#8b949e' }, ticks: { color: '#8b949e', font: { size: 10 } }, grid: { color: '#21262d' } },
      },
    },
  });
}

// -- Heatmap drawing --
function drawHeatmap(canvasId, data2d, xLabels, yLabels, colorFn) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const rows = data2d.length;
  const cols = data2d[0]?.length || 0;
  const cellW = Math.max(18, Math.min(40, 800 / cols));
  const cellH = Math.max(18, Math.min(30, 400 / rows));
  const marginL = 50, marginT = 40;
  canvas.width = marginL + cols * cellW;
  canvas.height = marginT + rows * cellH;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = data2d[r][c];
      ctx.fillStyle = colorFn(v);
      ctx.fillRect(marginL + c * cellW, marginT + r * cellH, cellW - 1, cellH - 1);
    }
  }
  // X labels
  ctx.fillStyle = '#8b949e';
  ctx.font = '9px monospace';
  ctx.textAlign = 'center';
  for (let c = 0; c < cols; c++) {
    const label = xLabels ? xLabels[c] : c;
    const text = typeof label === 'string' ? (label.length > 6 ? label.slice(0, 5) + '..' : label) : String(label);
    ctx.save();
    ctx.translate(marginL + c * cellW + cellW / 2, marginT - 4);
    ctx.rotate(-0.5);
    ctx.fillText(text, 0, 0);
    ctx.restore();
  }
  // Y labels
  ctx.textAlign = 'right';
  for (let r = 0; r < rows; r++) {
    ctx.fillText(yLabels ? yLabels[r] : `L${r}`, marginL - 4, marginT + r * cellH + cellH / 2 + 3);
  }
}

function heatColor(v, min, max) {
  const t = max > min ? (v - min) / (max - min) : 0;
  const r = Math.round(255 * Math.min(1, t * 2));
  const g = Math.round(255 * Math.min(1, t * 1.2));
  const b = Math.round(100 + 155 * (1 - t));
  return `rgb(${r},${g},${b})`;
}

function blueHeatColor(v) {
  const t = Math.max(0, Math.min(1, v));
  const r = Math.round(13 + 75 * t);
  const g = Math.round(17 + 149 * t);
  const b = Math.round(34 + 221 * t);
  return `rgb(${r},${g},${b})`;
}

// -- Render functions --
function renderOverview() {
  const layerLabels = Array.from({ length: N_LAYER }, (_, i) => `L${i}`);

  makeLineChart('chartSparsity', layerLabels,
    [{ label: 'L0 Sparsity', data: DATA.sparsity, borderColor: '#58a6ff', backgroundColor: '#58a6ff22', fill: true, tension: 0.3 }],
    'Fraction of zeros');

  const crLabels = ['Emb', ...layerLabels];
  makeLineChart('chartCodingRate', crLabels,
    [{ label: 'Coding Rate R(Z)', data: DATA.coding_rates, borderColor: '#3fb950', backgroundColor: '#3fb95022', fill: true, tension: 0.3 }],
    'R(Z)');

  // Sparsity heatmap
  if (DATA.odl_sparsity_per_token && DATA.odl_sparsity_per_token.length > 0) {
    drawHeatmap('canvasSparsityHeatmap', DATA.odl_sparsity_per_token, TOKENS,
      layerLabels, v => blueHeatColor(v));
  }
}

function renderThresholds() {
  if (!DATA.threshold_stats) return;
  const labels = [];
  const means = [], stds = [], mins = [], maxs = [];
  const tbody = document.querySelector('#thresholdTable tbody');

  DATA.threshold_stats.forEach((s, i) => {
    if (!s) return;
    labels.push(`L${i}`);
    means.push(s.mean);
    stds.push(s.std);
    mins.push(s.min);
    maxs.push(s.max);
    const row = document.createElement('tr');
    row.innerHTML = `<td>L${i}</td><td>${s.mean.toFixed(4)}</td><td>${s.std.toFixed(4)}</td><td>${s.min.toFixed(4)}</td><td>${s.median.toFixed(4)}</td><td>${s.max.toFixed(4)}</td>`;
    tbody.appendChild(row);
  });

  makeLineChart('chartThresholds', labels, [
    { label: 'Mean', data: means, borderColor: '#58a6ff', tension: 0.3 },
    { label: 'Min', data: mins, borderColor: '#3fb950', borderDash: [4, 4], tension: 0.3 },
    { label: 'Max', data: maxs, borderColor: '#f85149', borderDash: [4, 4], tension: 0.3 },
  ], 'Threshold value');
}

function renderScalars() {
  if (!DATA.layer_scalars) return;
  const labels = Array.from({ length: N_LAYER }, (_, i) => `L${i}`);
  makeBarChart('chartResidLambda', labels, DATA.layer_scalars.resid_lambdas, '#58a6ff', 'Lambda');
  makeBarChart('chartX0Lambda', labels, DATA.layer_scalars.x0_lambdas, '#d29922', 'Lambda');
}

function renderAttention() {
  const attn = DATA.attention_maps?.[selectedLayer];
  if (!attn || !attn[selectedHead]) return;
  const headData = attn[selectedHead]; // [T, T]
  drawHeatmap('canvasAttn', headData, TOKENS, TOKENS, v => blueHeatColor(v));
}

function renderDictionary() {
  const layerAtoms = DATA.odl_top_atoms?.[selectedLayer];
  if (!layerAtoms) return;

  // Build a heatmap: tokens x top-N atoms
  const allIndices = new Set();
  layerAtoms.forEach(ta => ta.indices.forEach(idx => allIndices.add(idx)));
  const sortedAtoms = [...allIndices].sort((a, b) => a - b).slice(0, 100);
  const atomToCol = {};
  sortedAtoms.forEach((a, i) => atomToCol[a] = i);

  const grid = [];
  for (let t = 0; t < TOKENS.length; t++) {
    const row = new Array(sortedAtoms.length).fill(0);
    const ta = layerAtoms[t];
    ta.indices.forEach((idx, j) => {
      if (atomToCol[idx] !== undefined) {
        row[atomToCol[idx]] = ta.values[j];
      }
    });
    grid.push(row);
  }

  if (grid.length > 0 && grid[0].length > 0) {
    const flat = grid.flat();
    const maxVal = Math.max(...flat.filter(v => v > 0), 1e-8);
    drawHeatmap('canvasDict', grid, sortedAtoms.map(a => `#${a}`), TOKENS,
      v => heatColor(v, 0, maxVal));
  }
}

function renderUtilization() {
  const util = DATA.atom_utilization?.[selectedLayer];
  if (!util) return;

  const statsEl = document.getElementById('utilStats');
  statsEl.innerHTML = `
    <div class="stat-box"><div class="value">${util.total_atoms}</div><div class="label">Total Atoms</div></div>
    <div class="stat-box"><div class="value">${util.active_atoms}</div><div class="label">Active Atoms</div></div>
    <div class="stat-box"><div class="value">${(100 * util.active_atoms / util.total_atoms).toFixed(1)}%</div><div class="label">Utilization</div></div>
    <div class="stat-box"><div class="value">${util.mean_activations.toFixed(1)}</div><div class="label">Mean Fires/Atom</div></div>
  `;

  makeBarChart('chartUtilization',
    util.top_50.map(a => `#${a}`),
    util.top_50_counts,
    '#d29922', 'Times activated');
}

function renderTokenDetail() {
  if (selectedToken < 0) return;
  document.getElementById('tokenDetailPlaceholder').style.display = 'none';
  document.getElementById('tokenDetailContent').style.display = 'block';
  document.getElementById('tokenDetailTitle').textContent =
    `Token ${selectedToken}: "${TOKENS[selectedToken]}"`;

  // Per-layer sparsity for this token
  const tokenSparsity = DATA.odl_sparsity_per_token?.map(layer => layer[selectedToken]) || [];
  const layerLabels = Array.from({ length: N_LAYER }, (_, i) => `L${i}`);

  makeLineChart('chartTokenSparsity', layerLabels,
    [{ label: 'L0 Sparsity', data: tokenSparsity, borderColor: '#58a6ff', backgroundColor: '#58a6ff22', fill: true, tension: 0.3 }],
    'Fraction of zeros');

  // Active atom count per layer
  const atomCounts = DATA.odl_top_atoms?.map(layer => layer[selectedToken]?.indices.length || 0) || [];
  makeBarChart('chartTokenAtoms', layerLabels, atomCounts, '#3fb950', 'Active atoms');

  // Stats
  const avgSparsity = tokenSparsity.length ? (tokenSparsity.reduce((a, b) => a + b, 0) / tokenSparsity.length) : 0;
  const avgAtoms = atomCounts.length ? (atomCounts.reduce((a, b) => a + b, 0) / atomCounts.length) : 0;
  document.getElementById('tokenStats').innerHTML = `
    <div class="stat-box"><div class="value">${(100 * avgSparsity).toFixed(1)}%</div><div class="label">Avg Sparsity</div></div>
    <div class="stat-box"><div class="value">${avgAtoms.toFixed(0)}</div><div class="label">Avg Active Atoms</div></div>
  `;

  renderTokenTopAtoms();
}

function renderTokenTopAtoms() {
  if (selectedToken < 0) return;
  const ta = DATA.odl_top_atoms?.[selectedLayer]?.[selectedToken];
  if (!ta || !ta.indices.length) return;

  const labels = ta.indices.slice(0, 50).map(i => `#${i}`);
  const values = ta.values.slice(0, 50);
  makeBarChart('chartTokenTopAtoms', labels, values, '#f85149', 'Activation magnitude');
}

// -- Boot --
document.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
