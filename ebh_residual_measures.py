#!/usr/bin/env python3
"""
EBH-family entropy measures on the residual stream: baseline vs beta.

Why
---
Attention-WEIGHT entropy is a bystander: beta's behavioral effect does not ride
on it. The value-mixing result predicts the effect lives in the residual-stream
activations. This script screens whether residual-stream COMPLEXITY measures
from the entropic-brain literature move under beta, and move MORE than
attention-weight entropy does.

Screening only (baseline vs beta). Establishes whether the measures respond to
beta at all and more than attention entropy. Does NOT establish behavioral
tracking -- that needs the A/B conditions. run_condition is structured so adding
conditions is trivial.

Measures (on residual stream = output hidden_states per layer)
--------------------------------------------------------------
  lzc            : Lempel-Ziv complexity of binarized activation sequence (per
                   unit thresholded at its median, concatenated, LZ76, normalized
                   by n/log2 n). EBH workhorse; temporal character.
  spatial_entropy: per-token Shannon entropy of across-unit |activation|
                   distribution, normalized by log(units); averaged over tokens.
                   NSC analog; no time axis, more robust.
  attn_entropy   : normalized attention-weight entropy at target layers only
                   (the NEGATIVE CONTROL). Captured cheaply via a hook on just the
                   target layers, NOT output_attentions over all layers.

PERF NOTE: this version does NOT use output_attentions=True (which forces the
model off its fast path and materializes attention for every layer). Attention
entropy is captured only at target layers via a lightweight forward hook, or
skipped entirely with --no-attn.

CAVEAT: LZc on token-indexed activations is an ANALOG of the EEG/MEG measure, not
identical; threshold choice affects magnitude. Spatial entropy is more robust;
prefer agreement between the two.

Reuses patch / prompts / generation from beta_psychedelic_sweep_llama.py.

Example
-------
  python ebh_residual_measures.py --beta 0.45 --layers 2,3 --num-prompts 8
  python ebh_residual_measures.py --beta 0.45 --layers 2,3 --num-prompts 8 --no-attn
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import beta_psychedelic_sweep_llama as core


# -----------------------------
# Complexity measures
# -----------------------------

def _lz76_clean(seq, max_window=1024) -> int:
    """Lempel-Ziv 1976 complexity with a BOUNDED back-search window.

    The textbook LZ76 searches the entire prefix for each substring, which is
    O(n^2) and intractable in pure Python on the million-bit strings produced by
    full hidden states. We bound the back-search to the last `max_window` bits.
    This slightly under-counts complexity on very long strings but is monotone,
    consistent across prompts, and turns the cost roughly linear. Input: 0/1."""
    s = [1 if b else 0 for b in seq]
    n = len(s)
    if n == 0:
        return 0
    c = 1
    l = 1
    k = 1
    while l + k <= n:
        sub = s[l:l + k]
        found = False
        lo = max(0, l - max_window)
        for start in range(lo, l):
            if s[start:start + k] == sub:
                found = True
                break
        if found:
            k += 1
        else:
            c += 1
            l += k
            k = 1
    return c


# Cap how many hidden units and tokens feed the LZc bit-string. Full hidden
# states make LZc intractable on CPU; a small fixed subsample keeps it fast AND
# comparable across prompts/layers. The bounded search window in _lz76_clean is
# the other lever; both are set conservatively for pure-Python CPU speed.
LZC_MAX_UNITS = 48
LZC_MAX_TOKENS = 192
_LZC_RNG = np.random.default_rng(0)


def lzc_normalized(activations: np.ndarray) -> float:
    """LZc of a [tokens, units] block, computed on a FIXED subsample of units and
    a capped token window so it's tractable on CPU. Binarize each sampled unit's
    trajectory at its median, concatenate, bounded-LZ76, normalize by n/log2 n."""
    T, U = activations.shape
    if T < 4:
        return float("nan")
    # cap tokens (take the most recent window -- the generated continuation)
    if T > LZC_MAX_TOKENS:
        activations = activations[-LZC_MAX_TOKENS:, :]
        T = LZC_MAX_TOKENS
    # deterministic unit subsample (same indices every call for comparability)
    if U > LZC_MAX_UNITS:
        idx = np.sort(_LZC_RNG.choice(U, size=LZC_MAX_UNITS, replace=False))
        activations = activations[:, idx]
        U = LZC_MAX_UNITS
    bits = []
    for u in range(U):
        col = activations[:, u]
        med = np.median(col)
        bits.extend((col > med).astype(int).tolist())
    c = _lz76_clean(bits)
    n = len(bits)
    norm = n / math.log2(n) if n > 1 else 1.0
    return float(c / norm)


def spatial_entropy(activations: np.ndarray) -> float:
    """Mean over tokens of normalized Shannon entropy of the across-unit
    |activation| distribution."""
    T, U = activations.shape
    if U < 2:
        return float("nan")
    a = np.abs(activations)
    row_sums = a.sum(axis=1, keepdims=True) + 1e-12
    p = a / row_sums
    ent = -(p * np.log(p + 1e-12)).sum(axis=1)
    return float((ent / math.log(U)).mean())


# -----------------------------
# Capture residual stream (+ cheap target-layer attention entropy)
# -----------------------------

@torch.no_grad()
def capture_for_prompt(model, tokenizer, text, max_tokens, attn_layers=None):
    """Forward returning hidden_states per layer [tokens, units] and (optionally)
    attention-weight entropy ONLY for attn_layers. No output_attentions=True."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens).to(model.device)

    attn_ent = {}
    handles = []
    if attn_layers:
        want = set(int(x) for x in attn_layers)

        def make_hook(idx):
            def hook(module, args, kwargs, output):
                if isinstance(output, tuple) and len(output) >= 2 and torch.is_tensor(output[1]):
                    p = output[1].float().clamp_min(1e-12)
                    e = -(p * p.log()).sum(dim=-1)
                    attn_ent[idx] = (e / math.log(max(p.shape[-1], 2))).mean().item()
                return output
            return hook

        for idx, layer in enumerate(model.model.layers):
            if idx in want:
                h = layer.self_attn.register_forward_hook(make_hook(idx), with_kwargs=True)
                handles.append(h)

    out = model(**enc, output_hidden_states=True, use_cache=False)
    for h in handles:
        h.remove()

    hs = [h[0].float().cpu().numpy() for h in out.hidden_states[1:]]  # skip embedding
    return hs, attn_ent


@torch.no_grad()
def run_condition(model, tokenizer, prompts, *, beta_ratio, beta_layers, target_layers,
                  gen_tokens, metric_max_tokens, label, capture_attn=True, full_lzc=False):
    rows = []
    attn_layers = target_layers if capture_attn else None
    for item in prompts:
        with core.beta_intervention(beta_ratio=beta_ratio, layers=beta_layers):
            output = core.generate(model, tokenizer, item["prompt"], gen_tokens,
                                    do_sample=False, temperature=0.0, seed=None)
            text = item["prompt"] + "\n" + output
            hs, attn_ent = capture_for_prompt(model, tokenizer, text, metric_max_tokens,
                                              attn_layers=attn_layers)

        drift = core.associative_drift(model, tokenizer, item["prompt"], output, metric_max_tokens)
        ppl = core.clean_perplexity(model, tokenizer, output, metric_max_tokens)
        local_coh = (1.0 / (1.0 + max(0.0, math.log(max(ppl, 1e-6)) - math.log(10.0)))
                     if not math.isnan(ppl) else float("nan"))

        n_layers = len(hs)
        tgt = [l for l in target_layers if 0 <= l < n_layers]

        def mean_over(layers, fn):
            vals = [fn(hs[l]) for l in layers]
            vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
            return float(np.mean(vals)) if vals else float("nan")

        rows.append({
            "condition": label,
            "prompt_id": item["prompt_id"],
            "associative_drift": drift,
            "perplexity": ppl,
            "local_coherence": local_coh,
            "lzc_target": mean_over(tgt, lzc_normalized),
            "spatial_entropy_target": mean_over(tgt, spatial_entropy),
            "attn_entropy_target": (float(np.mean([attn_ent[l] for l in tgt if l in attn_ent]))
                                    if (tgt and attn_ent) else float("nan")),
            # spatial entropy over all layers is cheap; LZc over all layers is the
            # main cost, so it is skipped unless --full-lzc is set (NaN otherwise).
            "lzc_mean": (mean_over(range(n_layers), lzc_normalized) if full_lzc else float("nan")),
            "spatial_entropy_mean": mean_over(range(n_layers), spatial_entropy),
            "attn_entropy_mean": (float(np.mean(list(attn_ent.values()))) if attn_ent else float("nan")),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-1B-Instruct"))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--outdir", default="ebh_residual")
    ap.add_argument("--beta", type=float, default=0.45)
    ap.add_argument("--layers", default="2,3", help="target layers beta acts on")
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gen-tokens", type=int, default=96)
    ap.add_argument("--metric-max-tokens", type=int, default=512)
    ap.add_argument("--no-attn", action="store_true",
                    help="skip attention-entropy control column entirely (fastest)")
    ap.add_argument("--full-lzc", action="store_true",
                    help="compute LZc over ALL layers (slow on CPU); default computes "
                         "it only at target layers, which is enough for the screen")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    target_layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {args.model} on {device} ({dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype,
        device_map=device if device != "cpu" else None,
        attn_implementation="eager",
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()
    core.patch_llama_attention()
    core.assert_patch_live(model, tokenizer)

    prompts = core.make_open_prompts(args.num_prompts, args.seed)
    capture_attn = not args.no_attn

    frames = []
    frames.append(run_condition(model, tokenizer, prompts,
                                beta_ratio=1.0, beta_layers=None, target_layers=target_layers,
                                gen_tokens=args.gen_tokens, metric_max_tokens=args.metric_max_tokens,
                                label="baseline", capture_attn=capture_attn, full_lzc=args.full_lzc))
    frames.append(run_condition(model, tokenizer, prompts,
                                beta_ratio=args.beta, beta_layers=target_layers, target_layers=target_layers,
                                gen_tokens=args.gen_tokens, metric_max_tokens=args.metric_max_tokens,
                                label=f"beta_{args.beta:g}", capture_attn=capture_attn, full_lzc=args.full_lzc))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(outdir / "per_prompt.csv", index=False)

    measure_cols = ["lzc_target", "spatial_entropy_target", "attn_entropy_target",
                    "lzc_mean", "spatial_entropy_mean", "attn_entropy_mean",
                    "associative_drift", "perplexity", "local_coherence"]
    agg = df.groupby("condition")[measure_cols].mean().reset_index()
    agg.to_csv(outdir / "summary_by_condition.csv", index=False)

    base = agg[agg["condition"] == "baseline"].iloc[0]
    beta = agg[agg["condition"].str.startswith("beta_")].iloc[0]

    def delta(col):
        return float(beta[col] - base[col])

    shifts = {c: delta(c) for c in
              ["lzc_target", "spatial_entropy_target", "attn_entropy_target",
               "lzc_mean", "spatial_entropy_mean", "attn_entropy_mean"]}

    have_attn = not math.isnan(shifts["attn_entropy_target"])
    if have_attn:
        attn_move = abs(shifts["attn_entropy_target"]) + 1e-9
        residual_moves_more = (abs(shifts["lzc_target"]) > 2 * attn_move or
                               abs(shifts["spatial_entropy_target"]) > 2 * attn_move)
    else:
        residual_moves_more = (abs(shifts["lzc_target"]) > 0.02 or
                               abs(shifts["spatial_entropy_target"]) > 0.02)

    verdict = {
        "behavioral_effect_present": {"drift_shift": delta("associative_drift"),
                                      "perplexity_shift": delta("perplexity")},
        "measure_shifts_baseline_to_beta": shifts,
        "attn_control_captured": bool(have_attn),
        "residual_measures_move_more_than_attn": bool(residual_moves_more),
        "interpretation": (
            "Residual-stream complexity moved more than attention entropy (or moved "
            "non-trivially when attn not captured). Necessary precondition met. NEXT: "
            "add A/B conditions to test behavioral TRACKING (move under beta/B where "
            "behavior moves, flat under A where it doesn't)."
            if residual_moves_more else
            "Residual measures did NOT clearly out-move attention entropy. Tune "
            "binarization/normalization or inspect per-layer values."),
        "caveats": [
            "Screening only (baseline vs beta); does NOT establish behavioral tracking.",
            "n=%d prompts; directional only." % args.num_prompts,
            "LZc on token-indexed activations is an analog, not identical; spatial "
            "entropy is more robust -- prefer agreement.",
        ],
    }

    summary = {"model": args.model, "beta": args.beta, "target_layers": target_layers,
               "num_prompts": args.num_prompts, "verdict": verdict,
               "files": {"per_prompt": "per_prompt.csv",
                         "summary_by_condition": "summary_by_condition.csv"}}
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\n=== EBH RESIDUAL-MEASURE SCREEN ===")
    print(json.dumps(summary, indent=2, default=float))
    print(f"\nWrote results to: {outdir.resolve()}")


if __name__ == "__main__":
    main()