"""
Apples-to-apples evaluation across pipeline stages.

Two modes:

  --mode core (default)
    Evaluates on 19 uncontaminated CORE tasks (log-prob based), excluding:
    - arc_easy, arc_challenge  (mid-train MMLU auxiliary_train contains ARC, chat SFT trains on ARC)
    - openbook_qa              (mid-train MMLU auxiliary_train contains OBQA)

  --mode chat
    Evaluates on the 2 uncontaminated chat tasks (MMLU + HumanEval).
    MMLU is categorical (fast), HumanEval is generative (slower).

Examples:

    python -m scripts.pipeline_eval --checkpoints base:d12 mid:d12 sft:d12
    python -m scripts.pipeline_eval --mode chat --checkpoints mid:d12 semisup_general:d12 sft:d12
    python -m scripts.pipeline_eval --mode chat --checkpoints mid:d12 sft:d12 --max-problems 200

With torchrun:
    torchrun --nproc_per_node=8 -m scripts.pipeline_eval -- --mode chat --checkpoints mid:d12 sft:d12
"""

import argparse
import gc
import os
from contextlib import nullcontext

import torch

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type
from nanochat.checkpoint_manager import load_model

# Tasks whose train data appears in mid-train or chat SFT
CONTAMINATED_TASKS = {"arc_easy", "arc_challenge", "openbook_qa"}

# Clean chat eval tasks (everything else is contaminated by mid-train/SFT/selfsup)
CLEAN_CHAT_TASKS = ["MMLU", "HumanEval"]
CHAT_BASELINES = {"MMLU": 0.25, "HumanEval": 0.0}

# Valid checkpoint sources (mirrors checkpoint_manager.load_model)
VALID_SOURCES = {
    "base", "mid", "sft", "rl",
    "semisup", "semisup_code", "semisup_general", "semisup_math",
    "selfflow",
}

# -----------------------------------------------------------------------------

def parse_checkpoint_spec(spec):
    """Parse 'source', 'source:tag', or 'source:tag:step'."""
    parts = spec.split(":")
    source = parts[0]
    if source not in VALID_SOURCES:
        raise ValueError(f"Unknown source '{source}'. Valid: {sorted(VALID_SOURCES)}")
    model_tag = parts[1] if len(parts) > 1 else None
    step = int(parts[2]) if len(parts) > 2 else None
    return source, model_tag, step


def format_label(source, meta, model_tag=None):
    """Human-readable label for a checkpoint."""
    step = meta.get("step", "?")
    if model_tag:
        return f"{source}:{model_tag} (step {step})"
    return f"{source} (step {step})"


def evaluate_chat_clean(model, tokenizer, engine, max_problems=None, batch_size=8):
    """Evaluate on the 2 clean chat tasks: MMLU (categorical) + HumanEval (generative)."""
    from scripts.chat_eval import run_chat_eval

    results = {}
    centered_results = {}
    for task_name in CLEAN_CHAT_TASKS:
        print0(f"Evaluating: {task_name}... ", end="", flush=True)
        acc = run_chat_eval(
            task_name, model, tokenizer, engine,
            batch_size=batch_size,
            num_samples=1,
            max_new_tokens=512,
            temperature=0.0,
            top_k=50,
            max_problems=max_problems,
        )
        baseline = CHAT_BASELINES[task_name]
        centered = (acc - baseline) / (1.0 - baseline)
        results[task_name] = acc
        centered_results[task_name] = centered
        print0(f"accuracy: {acc:.4f} | centered: {centered:.4f}")

    aggregate = sum(centered_results.values()) / len(centered_results)
    return {
        "results": results,
        "centered_results": centered_results,
        "core_metric": aggregate,  # reuse key name for unified table printing
    }


def print_comparison_table(all_results, labels, aggregate_name="CORE-clean"):
    """Print a side-by-side comparison table across checkpoints."""
    if not all_results:
        return

    task_labels = list(all_results[0]["centered_results"].keys())

    task_w = max(len(t) for t in task_labels + ["Task", aggregate_name]) + 2
    col_w = max(len(l) for l in labels) + 2
    col_w = max(col_w, 10)

    header = f"{'Task':<{task_w}}"
    for label in labels:
        header += f" | {label:>{col_w}}"
    sep = "-" * task_w + ("-+-" + "-" * col_w) * len(labels)

    print0("")
    print0("=" * len(sep))
    print0(f"{aggregate_name} Comparison (uncontaminated tasks only)")
    print0("=" * len(sep))
    print0(header)
    print0(sep)

    for task in task_labels:
        row = f"{task:<{task_w}}"
        for res in all_results:
            val = res["centered_results"].get(task)
            cell = f"{val:.4f}" if val is not None else "-"
            row += f" | {cell:>{col_w}}"
        print0(row)

    print0(sep)
    row = f"{aggregate_name:<{task_w}}"
    for res in all_results:
        row += f" | {res['core_metric']:>{col_w}.4f}"
    print0(row)
    print0("")


def write_csv(all_results, labels, output_path, aggregate_name="CORE-clean"):
    """Write a CSV with all results for easy import into spreadsheets."""
    task_labels = list(all_results[0]["centered_results"].keys())

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        f.write("Task," + ",".join(labels) + "\n")
        for task in task_labels:
            vals = []
            for res in all_results:
                v = res["centered_results"].get(task)
                vals.append(f"{v:.6f}" if v is not None else "")
            f.write(f"{task}," + ",".join(vals) + "\n")
        agg = [f"{res['core_metric']:.6f}" for res in all_results]
        f.write(f"{aggregate_name}," + ",".join(agg) + "\n")

    print0(f"Wrote comparison CSV to {output_path}")


# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Apples-to-apples evaluation across pipeline checkpoints"
    )
    parser.add_argument(
        "--checkpoints", type=str, nargs="+", required=True,
        help="Checkpoint specs: source or source:tag or source:tag:step "
             "(e.g. base semisup_code:d12 mid:d12 sft:d12:500)"
    )
    parser.add_argument(
        "--mode", type=str, default="core", choices=["core", "chat"],
        help="Evaluation mode: 'core' = 19 clean CORE tasks (log-prob), "
             "'chat' = MMLU + HumanEval (clean chat tasks)"
    )
    parser.add_argument(
        "--include-contaminated", action="store_true",
        help="[core mode] Also evaluate the 3 contaminated tasks (reported separately)"
    )
    parser.add_argument(
        "--max-per-task", type=int, default=-1,
        help="[core mode] Max examples per CORE task (-1 = all)"
    )
    parser.add_argument(
        "--max-problems", type=int, default=None,
        help="[chat mode] Max problems per chat task (default = all)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="[chat mode] Batch size for categorical tasks like MMLU"
    )
    parser.add_argument(
        "--output-csv", type=str, default=None,
        help="Path to write comparison CSV (default: auto-generated)"
    )
    args = parser.parse_args()

    # Distributed / precision setup
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device_type == "cuda" else nullcontext()
    )

    # Parse all checkpoint specs up front so we fail fast on bad input
    specs = [parse_checkpoint_spec(s) for s in args.checkpoints]

    aggregate_name = "CORE-clean" if args.mode == "core" else "Chat-clean"
    all_results = []
    all_contaminated_results = []
    labels = []

    for source, model_tag, step in specs:
        print0("")
        print0("=" * 70)
        print0(f"[{args.mode}] Evaluating checkpoint: {source}" +
               (f":{model_tag}" if model_tag else "") +
               (f":{step}" if step is not None else ""))
        print0("=" * 70)

        model, tokenizer, meta = load_model(
            source, device, phase="eval", model_tag=model_tag, step=step
        )
        label = format_label(source, meta, model_tag=model_tag)
        labels.append(label)

        with autocast_ctx:
            if args.mode == "core":
                from scripts.base_eval import evaluate_model
                out = evaluate_model(
                    model, tokenizer, device,
                    max_per_task=args.max_per_task,
                    exclude_tasks=CONTAMINATED_TASKS,
                )
            else:
                from nanochat.engine import Engine
                engine = Engine(model, tokenizer)
                out = evaluate_chat_clean(
                    model, tokenizer, engine,
                    max_problems=args.max_problems,
                    batch_size=args.batch_size,
                )

        all_results.append(out)
        print0(f"\n{label}  {aggregate_name} = {out['core_metric']:.4f}")

        # Optionally also evaluate contaminated CORE tasks (core mode only)
        if args.mode == "core" and args.include_contaminated:
            from scripts.base_eval import evaluate_model
            with autocast_ctx:
                full_out = evaluate_model(
                    model, tokenizer, device,
                    max_per_task=args.max_per_task,
                    exclude_tasks=set(),
                )
            contam_results = {}
            contam_centered = {}
            for t in CONTAMINATED_TASKS:
                if t in full_out["results"]:
                    contam_results[t] = full_out["results"][t]
                    contam_centered[t] = full_out["centered_results"][t]
            all_contaminated_results.append({
                "results": contam_results,
                "centered_results": contam_centered,
            })
            for t in sorted(CONTAMINATED_TASKS):
                if t in contam_centered:
                    print0(f"  [contaminated] {t}: {contam_centered[t]:.4f}")

        # Free memory before loading the next checkpoint
        del model
        gc.collect()
        if device_type == "cuda":
            torch.cuda.empty_cache()

    # Print comparison table
    print_comparison_table(all_results, labels, aggregate_name=aggregate_name)

    # Print contaminated comparison if requested
    if args.include_contaminated and all_contaminated_results:
        contam_tasks = sorted(CONTAMINATED_TASKS)
        task_w = max(len(t) for t in contam_tasks + ["Task"]) + 2
        col_w = max(len(l) for l in labels) + 2
        col_w = max(col_w, 10)
        sep = "-" * task_w + ("-+-" + "-" * col_w) * len(labels)
        print0("(Contaminated tasks — NOT included in CORE-clean)")
        print0(sep)
        header = f"{'Task':<{task_w}}"
        for label in labels:
            header += f" | {label:>{col_w}}"
        print0(header)
        print0(sep)
        for task in contam_tasks:
            row = f"{task:<{task_w}}"
            for res in all_contaminated_results:
                val = res["centered_results"].get(task)
                cell = f"{val:.4f}" if val is not None else "-"
                row += f" | {cell:>{col_w}}"
            print0(row)
        print0("")

    # Write CSV
    if ddp_rank == 0:
        output_csv = args.output_csv
        if output_csv is None:
            from datetime import datetime
            base_dir = get_base_dir()
            output_dir = os.path.join(base_dir, "pipeline_eval")
            os.makedirs(output_dir, exist_ok=True)
            suffix = "core" if args.mode == "core" else "chat"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sources = "_".join(s for s, _, _ in specs)
            tag = specs[0][1] or "default"
            tag_short = tag[:30]
            output_csv = os.path.join(output_dir, f"comparison_{suffix}_{sources}_{tag_short}_{timestamp}.csv")
        write_csv(all_results, labels, output_csv, aggregate_name=aggregate_name)

    # Log to report
    if ddp_rank == 0:
        from nanochat.report import get_report
        report_data = [{"mode": args.mode, "checkpoints": args.checkpoints}]
        for label, res in zip(labels, all_results):
            report_data.append({f"{label} {aggregate_name}": res["core_metric"]})
        get_report().log(section=f"Pipeline evaluation ({args.mode})", data=report_data)

    compute_cleanup()


if __name__ == "__main__":
    main()
