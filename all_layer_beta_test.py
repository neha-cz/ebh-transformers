#!/usr/bin/env python3
"""
All-layer attention-flattening test (no layer sweep).

Reuses the patch, generation, metrics, and analysis from
beta_psychedelic_sweep_llama.py, but:
  - flattens ALL layers (beta_layers=None), not an auto-selected subset
  - skips the slow one-layer-at-a-time sweep entirely
  - still runs the temperature control arm for comparison
  - ADDS intervened-layer-only attention-entropy reporting, so you can see
    whether all-layer flattening actually moves attention entropy (the 5-layer
    version barely did: global gap vs temperature was only ~0.014)

The main script must be in the same directory (or on PYTHONPATH) as this file,
under the name beta_psychedelic_sweep_llama.py.

Example
-------
    python all_layer_beta_test.py \
      --num-prompts 18 \
      --gen-tokens 96 \
      --beta-values 1.0,0.9,0.8,0.7,0.6,0.55,0.5,0.45,0.4,0.35,0.28,0.22,0.16,0.1 \
      --temp-values 0.7,1.0,1.3,1.6,2.0,2.5 \
      --outdir all_layer_results
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse everything from the main script so metrics stay byte-identical.
import beta_psychedelic_sweep_llama as core


def intervened_layer_entropy(results_df: pd.DataFrame, entropy_df: pd.DataFrame,
                             intervened_layers) -> pd.DataFrame:
    """Mean attention entropy restricted to the layers we actually flattened,
    per (arm, beta_ratio, temperature). With all-layer flattening this is every
    layer; the function still works and lets you compare against the temperature
    arm on the same layer set."""
    df = entropy_df.copy()
    if intervened_layers is not None:
        keep = set(int(x) for x in intervened_layers)
        df = df[df["attention_layer"].isin(keep)]
    grp = df.groupby(["beta_ratio", "temperature"], dropna=False)["attention_entropy"]
    out = grp.mean().reset_index().rename(columns={"attention_entropy": "intervened_layer_entropy"})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-1B-Instruct"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--outdir", default="all_layer_results")
    parser.add_argument("--num-prompts", type=int, default=18)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gen-tokens", type=int, default=96)
    parser.add_argument("--samples-per-prompt", type=int, default=3)
    parser.add_argument("--metric-max-tokens", type=int, default=1024)
    parser.add_argument("--beta-values",
                        default="1.0,0.9,0.8,0.7,0.6,0.55,0.5,0.45,0.4,0.35,0.28,0.22,0.16,0.1")
    parser.add_argument("--temp-values", default="0.7,1.0,1.3,1.6,2.0,2.5")
    parser.add_argument("--skip-temp-control", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32

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
    with open(outdir / "prompts.jsonl", "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    all_rows, all_entropy = [], []

    # ---- Baseline (no intervention, greedy) ----
    base_df, base_ent = core.evaluate_condition(
        model, tokenizer, prompts, "baseline",
        beta_ratio=1.0, layers=None, do_sample=False, temperature=0.0,
        samples_per_prompt=1, gen_tokens=args.gen_tokens,
        metric_max_tokens=args.metric_max_tokens, base_seed=args.seed,
    )
    all_rows.append(base_df)
    all_entropy.append(base_ent)
    baseline_summary = core.aggregate(base_df, ["condition"]).iloc[0].to_dict()

    n_layers = int(model.config.num_hidden_layers)

    # ---- ALL-LAYER beta sweep (greedy, vary beta, layers=None) ----
    beta_values = core.parse_floats(args.beta_values, ensure=1.0)
    for b in beta_values:
        df, ent = core.evaluate_condition(
            model, tokenizer, prompts, f"beta_{b:g}",
            beta_ratio=b, layers=None, do_sample=False, temperature=0.0,
            samples_per_prompt=1, gen_tokens=args.gen_tokens,
            metric_max_tokens=args.metric_max_tokens, base_seed=args.seed,
        )
        all_rows.append(df)
        all_entropy.append(ent)
    full_df = pd.concat(all_rows, ignore_index=True)
    beta_df = full_df[full_df["condition"].str.startswith("beta_")].copy()
    beta_summary = core.aggregate(beta_df, ["beta_ratio", "layers"])
    beta_summary.to_csv(outdir / "beta_summary.csv", index=False)
    band = core.detect_interesting_band(beta_summary, baseline_summary)

    # ---- Temperature control (beta=1.0, vary sampling temperature) ----
    temp_summary = pd.DataFrame()
    if not args.skip_temp_control:
        temp_values = core.parse_floats(args.temp_values, reverse=False)
        for t in temp_values:
            df, ent = core.evaluate_condition(
                model, tokenizer, prompts, f"temp_{t:g}",
                beta_ratio=1.0, layers=None, do_sample=True, temperature=t,
                samples_per_prompt=args.samples_per_prompt, gen_tokens=args.gen_tokens,
                metric_max_tokens=args.metric_max_tokens, base_seed=args.seed,
            )
            all_rows.append(df)
            all_entropy.append(ent)
        full_df = pd.concat(all_rows, ignore_index=True)
        temp_df = full_df[full_df["condition"].str.startswith("temp_")].copy()
        temp_summary = core.aggregate(temp_df, ["temperature"])
        temp_summary.to_csv(outdir / "temp_summary.csv", index=False)

    control = core.compare_beta_vs_temperature(beta_summary, temp_summary)

    # ---- Intervened-layer-only entropy (here = all layers) ----
    entropy_df = pd.concat(all_entropy, ignore_index=True)
    entropy_df.to_csv(outdir / "per_layer_attention_entropy.csv", index=False)
    il_entropy = intervened_layer_entropy(full_df, entropy_df, intervened_layers=None)
    il_entropy.to_csv(outdir / "intervened_layer_entropy.csv", index=False)

    # Headline entropy comparison on the intervened layer set.
    beta_il = il_entropy[il_entropy["temperature"] == 0.0]
    temp_il = il_entropy[il_entropy["temperature"] != 0.0]
    il_gap = None
    if not beta_il.empty and not temp_il.empty:
        il_gap = float(beta_il["intervened_layer_entropy"].max()
                       - temp_il["intervened_layer_entropy"].max())

    # ---- Persist ----
    results_df = pd.concat(all_rows, ignore_index=True)
    results_df.to_csv(outdir / "per_generation_results.csv", index=False)
    core.make_plots(pd.DataFrame(), beta_summary, temp_summary, outdir)

    summary = {
        "model": args.model,
        "n_layers": n_layers,
        "intervention": "ALL layers flattened (no subset)",
        "num_prompts": args.num_prompts,
        "gen_tokens": args.gen_tokens,
        "samples_per_prompt": args.samples_per_prompt,
        "patch_calls": core.PATCH_CALLS,
        "baseline": baseline_summary,
        "interesting_band": band,
        "beta_vs_temperature_control": control,
        "intervened_layer_entropy_gap_beta_minus_temp": il_gap,
        "intervened_layer_entropy_note": (
            "max intervened-layer attention entropy, beta arm minus temperature arm. "
            "Positive and meaningfully > the ~0.014 you got with 5 layers means all-layer "
            "flattening moves the attention landscape more than temperature does."
        ),
        "files": {
            "prompts": "prompts.jsonl",
            "per_generation_results": "per_generation_results.csv",
            "per_layer_attention_entropy": "per_layer_attention_entropy.csv",
            "intervened_layer_entropy": "intervened_layer_entropy.csv",
            "beta_summary": "beta_summary.csv",
            "temp_summary": "temp_summary.csv" if not temp_summary.empty else None,
        },
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, default=float))
    print(f"\nWrote results to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
