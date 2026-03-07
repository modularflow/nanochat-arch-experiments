"""
Semi-supervised self-training via iterative pseudo-labeling.

Provides utilities for:
1. Loading unlabeled prompts from JSONL files or existing Task objects
2. Generating candidate responses with the Engine
3. Scoring candidates by model confidence (average log-probability)
4. Filtering candidates by threshold, top-k, or task reward
5. Wrapping filtered pseudo-labels into a dataset for SFT-style training
"""

import json
import copy
import torch
import torch.nn.functional as F
from contextlib import nullcontext

from nanochat.common import print0, get_dist_info


class PromptSource:
    """
    Loads user prompts from a JSONL file or an existing Task instance.

    JSONL format: each line is {"messages": [{"role": "user", "content": "..."}]}
    Task mode: extracts the user turn(s) from each conversation, dropping assistant turns.
    """

    def __init__(self, filepath=None, task=None):
        assert (filepath is None) != (task is None), "Provide exactly one of filepath or task"
        self.prompts = []
        if filepath is not None:
            self._load_jsonl(filepath)
        else:
            self._load_task(task)

    def _load_jsonl(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                assert "messages" in obj, f"Each JSONL line must have a 'messages' key, got: {list(obj.keys())}"
                self.prompts.append(obj)

    def _load_task(self, task):
        for i in range(len(task)):
            conversation = copy.deepcopy(task[i])
            messages = conversation["messages"]
            # Keep only user turns (strip assistant responses)
            user_messages = [m for m in messages if m["role"] == "user"]
            assert len(user_messages) >= 1, f"Task example {i} has no user messages"
            # Store the full conversation for reward-based filtering later,
            # but mark which messages are user-only prompts
            self.prompts.append({
                "messages": [user_messages[0]],
                "_full_conversation": conversation,
            })

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        return self.prompts[index]


@torch.no_grad()
def generate_pseudo_labels(engine, prompt_source, tokenizer, num_candidates=4,
                           max_new_tokens=256, temperature=1.0, top_k=50,
                           device_batch_size=4, device="cuda", autocast_ctx=None):
    """
    Generate candidate responses for each prompt.

    Returns a list of dicts, one per prompt:
        {
            "prompt": <original prompt dict>,
            "candidates": [
                {"tokens": [...], "text": "...", "response_tokens": [...]},
                ...
            ]
        }
    """
    if autocast_ctx is None:
        autocast_ctx = nullcontext()

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    model = engine.model
    model.eval()

    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    bos = tokenizer.get_bos_token_id()

    results = []
    # Each rank handles a shard of the prompts
    for idx in range(ddp_rank, len(prompt_source), ddp_world_size):
        prompt = prompt_source[idx]

        # Build a prompt-only conversation and render it for completion
        prompt_conversation = copy.deepcopy(prompt)
        msgs = prompt_conversation["messages"]
        # Add a dummy assistant message so render_for_completion can pop it
        msgs.append({"role": "assistant", "content": ""})
        tokens = tokenizer.render_for_completion(prompt_conversation)
        prefix_length = len(tokens)

        # Generate candidates in batches to avoid OOM
        all_sequences = []
        num_passes = (num_candidates + device_batch_size - 1) // device_batch_size
        for pass_idx in range(num_passes):
            samples_this_pass = min(device_batch_size, num_candidates - len(all_sequences))
            seed = hash((idx, pass_idx)) & 0x7FFFFFFF
            with autocast_ctx:
                seqs, _masks = engine.generate_batch(
                    tokens,
                    num_samples=samples_this_pass,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    seed=seed,
                )
            all_sequences.extend(seqs)

        candidates = []
        for seq in all_sequences:
            response_tokens = seq[prefix_length:]
            response_text = tokenizer.decode(response_tokens) if response_tokens else ""
            candidates.append({
                "tokens": seq,
                "text": response_text,
                "response_tokens": response_tokens,
            })

        results.append({
            "prompt_idx": idx,
            "prompt": prompt,
            "candidates": candidates,
        })

        if (len(results) % 10) == 0:
            print0(f"  Generated candidates for {len(results)} prompts so far...")

    print0(f"  Generated candidates for {len(results)} prompts total on this rank.")
    return results


@torch.no_grad()
def score_by_confidence(model, tokenizer, pseudo_labels, device="cuda",
                        autocast_ctx=None, batch_size=16):
    """
    Score each candidate by average log-probability of its generated tokens.

    Modifies each candidate dict in-place, adding a "confidence" key.
    Higher confidence = model assigns higher probability to the response.
    Processes candidates in batches for throughput.
    """
    if autocast_ctx is None:
        autocast_ctx = nullcontext()

    model.eval()
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")

    # Flatten all candidates into a single list for batched scoring
    flat_cands = []
    for item in pseudo_labels:
        for cand in item["candidates"]:
            if len(cand["response_tokens"]) == 0:
                cand["confidence"] = float("-inf")
            else:
                flat_cands.append(cand)

    # Process in batches
    for batch_start in range(0, len(flat_cands), batch_size):
        batch_cands = flat_cands[batch_start:batch_start + batch_size]
        seqs = [c["tokens"] for c in batch_cands]
        max_len = max(len(s) for s in seqs)

        # Pad sequences to equal length
        padded_seqs = [s + [pad_token_id] * (max_len - len(s)) for s in seqs]
        ids = torch.tensor(padded_seqs, dtype=torch.long, device=device)
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone()

        # Mask out padding positions
        for i, seq in enumerate(seqs):
            if len(seq) - 1 < targets.size(1):
                targets[i, len(seq) - 1:] = -1

        with autocast_ctx:
            loss_per_token = model(inputs, targets, loss_reduction='none')
        loss_per_token = loss_per_token.view(inputs.size(0), inputs.size(1))

        # Extract per-candidate average log-prob over response tokens only
        for i, cand in enumerate(batch_cands):
            seq_len = len(cand["tokens"])
            resp_len = len(cand["response_tokens"])
            prefix_len = seq_len - resp_len - 1
            response_losses = loss_per_token[i, prefix_len:seq_len - 1]
            cand["confidence"] = -response_losses.mean().item()

        if (batch_start // batch_size) % 10 == 0:
            print0(f"  Scored {min(batch_start + batch_size, len(flat_cands))}/{len(flat_cands)} candidates...")

    return pseudo_labels


def filter_candidates(pseudo_labels, strategy="top_k", threshold_percentile=75,
                      top_k=1, task=None):
    """
    Filter candidates and return a list of conversation dicts ready for SFT training.

    Strategies:
        "top_k"     - keep the top_k highest-confidence candidates per prompt
        "threshold"  - keep candidates above the given confidence percentile
        "reward"     - use task.reward() to keep only correct/positive candidates

    Returns a list of {"messages": [...]} conversation dicts.
    """
    filtered = []

    if strategy == "reward":
        assert task is not None, "reward strategy requires a task with a reward() method"
        for item in pseudo_labels:
            prompt = item["prompt"]
            full_conv = prompt.get("_full_conversation")
            if full_conv is None:
                continue
            for cand in item["candidates"]:
                reward = task.reward(copy.deepcopy(full_conv), cand["text"])
                if reward > 0:
                    filtered.append(_build_conversation(prompt, cand["text"]))
    elif strategy == "top_k":
        for item in pseudo_labels:
            sorted_cands = sorted(item["candidates"], key=lambda c: c.get("confidence", float("-inf")), reverse=True)
            for cand in sorted_cands[:top_k]:
                if cand.get("confidence", float("-inf")) > float("-inf"):
                    filtered.append(_build_conversation(item["prompt"], cand["text"]))
    elif strategy == "threshold":
        all_scores = [c.get("confidence", float("-inf"))
                      for item in pseudo_labels
                      for c in item["candidates"]
                      if c.get("confidence", float("-inf")) > float("-inf")]
        if not all_scores:
            return filtered
        all_scores.sort()
        cutoff_idx = int(len(all_scores) * threshold_percentile / 100.0)
        cutoff_idx = min(cutoff_idx, len(all_scores) - 1)
        cutoff = all_scores[cutoff_idx]
        for item in pseudo_labels:
            for cand in item["candidates"]:
                if cand.get("confidence", float("-inf")) >= cutoff:
                    filtered.append(_build_conversation(item["prompt"], cand["text"]))
    else:
        raise ValueError(f"Unknown filter strategy: {strategy}")

    return filtered


def _build_conversation(prompt, response_text):
    """Build a standard conversation dict from a prompt and a generated response string."""
    user_messages = [m for m in prompt["messages"] if m["role"] == "user"]
    messages = []
    for m in user_messages:
        messages.append({"role": "user", "content": m["content"]})
        # Only one assistant response per user turn for now
        break
    messages.append({"role": "assistant", "content": response_text})
    return {"messages": messages}


class PseudoLabelDataset:
    """
    Wraps a list of filtered conversation dicts into a dataset compatible
    with the SFT data generator pattern.

    Supports __len__ and __getitem__ so it can be used like a Task.
    """

    def __init__(self, conversations):
        self.conversations = conversations

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, index):
        return self.conversations[index]
