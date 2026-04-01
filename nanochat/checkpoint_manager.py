"""
Utilities for saving and loading model/optim/state checkpoints.
"""
import os
import re
import glob
import json
import logging
import torch

from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)

def _patch_missing_config_keys(model_config_kwargs):
    """Add default values for new config keys missing in old checkpoints."""
    # Old models were trained with full context (no sliding window)
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"
        log0(f"Patching missing window_pattern in model config to 'L'")

def _patch_missing_keys(model_data, model_config):
    """Add default values for new parameters that may be missing in old checkpoints."""
    n_layer = model_config.n_layer
    # resid_lambdas defaults to 1.0 (identity scaling)
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(n_layer)
        log0(f"Patching missing resid_lambdas in model data to 1.0")
    # x0_lambdas defaults to 0.0 (disabled)
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(n_layer)
        log0(f"Patching missing x0_lambdas in model data to 0.0")

def _uses_selfflow_architecture(model_data):
    # SelfFlowCRATE checkpoints have proj_heads and/or corruption_conditioner keys.
    return any(k.startswith("proj_heads.") or k.startswith("corruption_conditioner.") for k in model_data.keys())

def _uses_crate_architecture(model_data):
    # CRATE checkpoints use mssa/(odl|ista) module names instead of attn/mlp.
    return any(".mssa." in key or ".odl." in key or ".ista." in key for key in model_data.keys())

def _uses_trm_gpt_architecture(model_data):
    return 'trm_marker' in model_data

def _uses_rys_gpt_architecture(model_data):
    return 'rys_layer_map' in model_data

def _uses_noq_gpt_architecture(model_data):
    # No-Q GPT has c_k and c_v but no c_q in attention blocks.
    has_ck = any(".c_k." in key for key in model_data.keys())
    has_cq = any(".c_q." in key for key in model_data.keys())
    has_mssa = any(".mssa." in key or ".kv." in key for key in model_data.keys())
    return has_ck and not has_cq and not has_mssa

def _uses_noq_crate_architecture(model_data):
    # No-Q CRATE uses .mssa.kv. (not .mssa.qkv.) alongside ODL/ISTA blocks.
    has_mssa_kv = any(".mssa.kv." in key for key in model_data.keys())
    has_mssa_qkv = any(".mssa.qkv." in key for key in model_data.keys())
    return has_mssa_kv and not has_mssa_qkv

def _detect_sparse_block_type(model_data):
    # Newer CRATE-alpha checkpoints store ODL parameters.
    if any(".odl." in key for key in model_data.keys()):
        return "odl"
    # Legacy CRATE checkpoints used ISTA weights.
    if any(".ista." in key for key in model_data.keys()):
        return "ista"
    return "odl"

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the model state parameters
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        logger.info(f"Saved model parameters to: {model_path}")
        # Save the metadata dict as json
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved metadata to: {meta_path}")
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if optimizer_data is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        torch.save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer state to: {optimizer_path}")

def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    _patch_missing_config_keys(model_config_kwargs)
    # Fix vocab_size if the metadata disagrees with the actual embedding weights.
    # This can happen when mid_train.py saves tokenizer.get_vocab_size() instead
    # of model.config.vocab_size.
    wte_key = next((k for k in model_data if k.endswith("wte.weight")), None)
    if wte_key is not None:
        actual_vocab = model_data[wte_key].shape[0]
        if model_config_kwargs.get("vocab_size") != actual_vocab:
            log0(f"Patching vocab_size: meta says {model_config_kwargs.get('vocab_size')}, weights say {actual_vocab}")
            model_config_kwargs["vocab_size"] = actual_vocab
    log0(f"Building model with config: {model_config_kwargs}")
    if _uses_selfflow_architecture(model_data):
        from nanochat.self_flow_model import SelfFlowCRATE, SelfFlowConfig
        model_class, model_config_class = SelfFlowCRATE, SelfFlowConfig
        arch_name = "SelfFlowCRATE"
        sparse_block_type = _detect_sparse_block_type(model_data)
        model_config_kwargs.setdefault("sparse_block_type", sparse_block_type)
        # Pull self-flow config from metadata if available
        selfflow_config = meta_data.get("selfflow_config", {})
        for sf_key in ("student_layers", "teacher_layers", "proj_hidden_mult",
                       "corruption_conditioning", "cond_hidden_mult",
                       "corruption_strategy", "rep_loss_type", "rep_loss_weight"):
            if sf_key in selfflow_config:
                model_config_kwargs.setdefault(sf_key, selfflow_config[sf_key])
        log0(f"Detected SelfFlowCRATE (sparse block: {sparse_block_type})")
    elif _uses_noq_crate_architecture(model_data):
        from nanochat.noq_crate import NoQCRATE, NoQCRATEConfig
        model_class, model_config_class = NoQCRATE, NoQCRATEConfig
        arch_name = "NoQCRATE"
        sparse_block_type = _detect_sparse_block_type(model_data)
        model_config_kwargs.setdefault("sparse_block_type", sparse_block_type)
        log0(f"Detected NoQ-CRATE sparse block type: {sparse_block_type}")
    elif _uses_crate_architecture(model_data):
        from nanochat.crate import CRATE, CRATEConfig
        model_class, model_config_class = CRATE, CRATEConfig
        arch_name = "CRATE"
        sparse_block_type = _detect_sparse_block_type(model_data)
        model_config_kwargs.setdefault("sparse_block_type", sparse_block_type)
        log0(f"Detected CRATE sparse block type: {sparse_block_type}")
    elif _uses_trm_gpt_architecture(model_data):
        from nanochat.trm_gpt import TRMGPT, TRMGPTConfig
        model_class, model_config_class = TRMGPT, TRMGPTConfig
        arch_name = "TRMGPT"
    elif _uses_rys_gpt_architecture(model_data):
        from nanochat.rys_gpt import RYSGPT, RYSGPTConfig
        model_class, model_config_class = RYSGPT, RYSGPTConfig
        arch_name = "RYSGPT"
    elif _uses_noq_gpt_architecture(model_data):
        from nanochat.noq_gpt import NoQGPT, NoQGPTConfig
        model_class, model_config_class = NoQGPT, NoQGPTConfig
        arch_name = "NoQGPT"
    else:
        model_class, model_config_class = GPT, GPTConfig
        arch_name = "GPT"
    log0(f"Detected checkpoint architecture: {arch_name}")
    model_config = model_config_class(**model_config_kwargs)
    _patch_missing_keys(model_data, model_config)
    with torch.device("meta"):
        model = model_class(model_config)
    # Load the model state
    model.to_empty(device=device)
    model.init_weights() # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # Load the Tokenizer
    tokenizer = get_tokenizer()
    # Sanity checks: compatibility between model and tokenizer
    tokenizer_vocab_size = tokenizer.get_vocab_size()
    model_vocab_size = model_config_kwargs["vocab_size"]
    if tokenizer_vocab_size > model_vocab_size:
        # CRATE models are trained with a smaller vocab than the default tokenizer.
        # This is fine: the model clips logits to its own vocab_size and the
        # tokenizer's decode() filters out-of-range IDs.
        log0(
            f"Tokenizer vocab ({tokenizer_vocab_size}) is larger than model vocab ({model_vocab_size}); "
            "tokens above model vocab will be unused (normal for CRATE)."
        )
    if tokenizer_vocab_size < model_vocab_size:
        log0(
            f"Tokenizer vocab ({tokenizer_vocab_size}) is smaller than model vocab ({model_vocab_size}); "
            "continuing with tokenizer-limited coverage."
        )
    return model, tokenizer, meta_data


def find_largest_model(checkpoints_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data

def load_model(source, *args, **kwargs):
    model_dir = {
        "base": "base_checkpoints",
        "mid": "mid_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
        "semisup": "semisup_checkpoints",
        "semisup_code": "semisup_code_checkpoints",
        "semisup_general": "semisup_general_checkpoints",
        "semisup_math": "semisup_math_checkpoints",
        "selfflow": "selfflow_checkpoints",
        "selfflow_pretrain": "selfflow_pretrain_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)
