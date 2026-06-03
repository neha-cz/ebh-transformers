#!/usr/bin/env python3
"""
Annealing beta-schedule screen.

Question
--------
Static beta sits at a fixed point in drift x coherence space. Simulated annealing
(which REBUS explicitly invokes) predicts that a COOLING SCHEDULE -- start hot/low-beta
(explore associations), end cold/high-beta (commit coherently) -- reaches a
DIFFERENT end state than any static beta. The hoped-for state is "lucid but loose":
high associative drift WITH preserved coherence, which neither static low-beta
(loose but broken) nor static high-beta (sharp but literal) achieves.

This is a SCREEN, not a causal study. It answers one decisive question:
  Does any cooling schedule land in a region of drift x coherence space that NO
  static beta value reaches?
    - If yes  -> scheduling accesses states static flattening cannot -> worth a
                 causal follow-up.
    - If no   -> the schedule is equivalent to some average static beta -> keep it
                 as a product feature, no new science.

Key implementation point
-------------------------
A cooling schedule requires beta to CHANGE during generation (different beta at
token 10 vs token 60). core.generate runs the whole decode loop with beta fixed,
so we cannot use it. Instead we run a MANUAL token-by-token decode loop and mutate
core.INTERVENTION.beta_ratio before each step according to the schedule. The patch
reads beta_ratio live on every forward pass, so per-step mutation takes effect.

Autoregressive caveat (worth stating regardless of result)
----------------------------------------------------------
Annealing in optimization revisits the whole state as it cools. Autoregressive
generation cannot: tokens emitted while hot are FIXED -- later cooling can only
sharpen the tokens still to come, not re-cohere what was already produced. So the
explore-then-commit dynamic may not transfer. If schedules fail to reach a new
region, this asymmetry is the likely reason and is itself a reportable observation.

Example
-------
  python annealing_beta_schedule.py --layers 2,3 --num-prompts 6
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
# Schedules: map generation progress t in [0,1] -> beta ratio
# -----------------------------

def make_schedules(beta_lo: float, beta_hi: float):
    """Return {name: fn(t)->beta_ratio} for t in [0,1] (0=start, 1=end of gen)."""
    lo, hi = beta_lo, beta_hi
    return {
        # cooling: hot (low beta, explore) -> cold (high beta, commit)
        "cool_linear":   lambda t: lo + (hi - lo) * t,
        "cool_exp":      lambda t: lo + (hi - lo) * (1.0 - math.exp(-3.0 * t)) / (1.0 - math.exp(-3.0)),
        "cool_late":     lambda t: lo + (hi - lo) * (t ** 2),          # stay hot, sharpen late
        "cool_early":    lambda t: lo + (hi - lo) * math.sqrt(t),      # sharpen early
        # warming (control: cold -> hot) -- should be worse if cooling is special
        "warm_linear":   lambda t: hi + (lo - hi) * t,
    }


@torch.no_grad()
def generate_scheduled(model, tokenizer, prompt, gen_tokens, *, layers,
                       schedule_fn, static_ratio=None, seed=None):
    """Manual greedy decode. Before each step, set core.INTERVENTION.beta_ratio
    from the schedule (or a fixed static_ratio). Returns decoded continuation."""
    if seed is not None:
        torch.manual_seed(seed)

    msgs = [{"role": "user", "content": prompt}]
    enc = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                        return_tensors="pt", return_dict=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    input_ids = enc["input_ids"]
    attn = enc.get("attention_mask")

    eos_ids = {tokenizer.eos_token_id}
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot is not None and eot != tokenizer.unk_token_id:
        eos_ids.add(eot)

    layers_set = set(layers)
    core.INTERVENTION.active = True
    core.INTERVENTION.layers = layers_set

    past = None
    generated = []
    cur_ids = input_ids
    cur_attn = attn
    for step in range(gen_tokens):
        t = step / max(gen_tokens - 1, 1)
        ratio = static_ratio if static_ratio is not None else schedule_fn(t)
        core.INTERVENTION.beta_ratio = float(ratio)

        out = model(input_ids=cur_ids, attention_mask=cur_attn,
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_logits = out.logits[:, -1, :]
        next_id = int(next_logits.argmax(dim=-1).item())
        generated.append(next_id)
        if next_id in eos_ids:
            break
        cur_ids = torch.tensor([[next_id]], device=model.device)
        if cur_attn is not None:
            cur_attn = torch.cat([cur_attn, torch.ones((1, 1), dtype=cur_attn.dtype,
                                                        device=model.device)], dim=1)

    core.INTERVENTION.active = False
    core.INTERVENTION.beta_ratio = 1.0
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.no_grad()
def run_condition(model, tokenizer, prompts, label, *, layers, gen_tokens,
                  metric_max_tokens, schedule_fn=None, static_ratio=None):
    rows = []
    for item in prompts:
        output = generate_scheduled(model, tokenizer, item["prompt"], gen_tokens,
                                    layers=layers, schedule_fn=schedule_fn,
                                    static_ratio=static_ratio, seed=None)
        drift = core.associative_drift(model, tokenizer, item["prompt"], output, metric_max_tokens)
        ppl = core.clean_perplexity(model, tokenizer, output, metric_max_tokens)
        coh = (1.0 / (1.0 + max(0.0, math.log(max(ppl, 1e-6)) - math.log(10.0)))
               if not math.isnan(ppl) else float("nan"))
        rows.append({"condition": label, "prompt_id": item["prompt_id"],
                     "associative_drift": drift, "perplexity": ppl, "local_coherence": coh})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-1B-Instruct"))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--outdir", default="annealing_screen")
    ap.add_argument("--layers", default="2,3")
    ap.add_argument("--beta-lo", type=float, default=0.45, help="hot end (explore)")
    ap.add_argument("--beta-hi", type=float, default=1.0, help="cold end (commit)")
    ap.add_argument("--static-grid", default="0.45,0.55,0.65,0.75,0.85,1.0",
                    help="static beta values to map the baseline frontier")
    ap.add_argument("--num-prompts", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gen-tokens", type=int, default=80)
    ap.add_argument("--metric-max-tokens", type=int, default=256)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    static_grid = [float(x) for x in args.static_grid.split(",") if x.strip()]

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
        attn_implementation="eager")
    if device == "cpu":
        model = model.to(device)
    model.eval()
    core.patch_llama_attention()
    core.assert_patch_live(model, tokenizer)

    prompts = core.make_open_prompts(args.num_prompts, args.seed)
    schedules = make_schedules(args.beta_lo, args.beta_hi)

    frames = []
    # static frontier
    for b in static_grid:
        frames.append(run_condition(model, tokenizer, prompts, f"static_{b:g}",
                                    layers=layers, gen_tokens=args.gen_tokens,
                                    metric_max_tokens=args.metric_max_tokens,
                                    static_ratio=b))
        print(f"[static {b:g}] done")
    # schedules
    for name, fn in schedules.items():
        frames.append(run_condition(model, tokenizer, prompts, name,
                                    layers=layers, gen_tokens=args.gen_tokens,
                                    metric_max_tokens=args.metric_max_tokens,
                                    schedule_fn=fn))
        print(f"[schedule {name}] done")

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(outdir / "per_prompt.csv", index=False)
    agg = df.groupby("condition").agg(
        drift=("associative_drift", "mean"),
        coherence=("local_coherence", "mean"),
        perplexity=("perplexity", "mean"),
        n=("perplexity", "count")).reset_index()
    agg.to_csv(outdir / "summary.csv", index=False)

    # ---- decisive test: does any schedule reach a region no static beta does? ----
    statics = agg[agg["condition"].str.startswith("static_")]
    scheds = agg[~agg["condition"].str.startswith("static_")]

    # Define the "lucid-loose" target: drift >= max static drift achieved at
    # coherence >= a threshold. We look for schedules that DOMINATE the static
    # frontier: higher drift than any static condition AT EQUAL-OR-BETTER coherence.
    def dominates_frontier(row):
        # is there NO static condition with both >= drift and >= coherence?
        better = statics[(statics["drift"] >= row["drift"] - 1e-9) &
                         (statics["coherence"] >= row["coherence"] - 1e-9)]
        # exclude trivial self; a schedule is frontier-beating if no static point
        # is at least as good on BOTH axes
        return len(better) == 0

    scheds = scheds.copy()
    scheds["beats_static_frontier"] = scheds.apply(dominates_frontier, axis=1)
    winners = scheds[scheds["beats_static_frontier"]]

    # also report: best coherence among conditions with drift above the static median
    static_drift_max = float(statics["drift"].max())
    high_drift = agg[agg["drift"] >= static_drift_max - 1e-9]

    verdict = {
        "static_frontier": statics[["condition", "drift", "coherence", "perplexity"]].to_dict("records"),
        "schedules": scheds[["condition", "drift", "coherence", "perplexity",
                             "beats_static_frontier"]].to_dict("records"),
        "any_schedule_beats_static_frontier": bool(len(winners) > 0),
        "frontier_beating_schedules": winners["condition"].tolist(),
        "interpretation": (
            "At least one cooling schedule reached a (drift, coherence) point that "
            "NO static beta dominates -- scheduling accesses a state static "
            "flattening cannot. This is a real phenomenon; a causal follow-up is "
            "warranted to explain WHY the trajectory matters."
            if len(winners) > 0 else
            "No schedule beat the static frontier: every schedule's (drift, coherence) "
            "is matched-or-dominated by some static beta. The schedule behaves like an "
            "effective average beta -> keep as a product feature, no new science. "
            "Likely cause: autoregressive generation fixes hot-phase tokens, so later "
            "cooling cannot re-cohere them (the explore-then-commit dynamic does not "
            "transfer from optimization to fixed-past decoding)."),
        "caveats": ["Screen only, greedy decode, n=%d prompts." % args.num_prompts,
                    "Frontier test is on mean drift/coherence; per-prompt variance not modeled."],
    }
    summary = {"model": args.model, "layers": layers,
               "beta_lo": args.beta_lo, "beta_hi": args.beta_hi,
               "static_grid": static_grid, "num_prompts": args.num_prompts,
               "verdict": verdict,
               "files": {"per_prompt": "per_prompt.csv", "summary": "summary.csv"}}
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\n=== ANNEALING SCHEDULE SCREEN ===")
    print(json.dumps(summary, indent=2, default=float))
    print(f"\nWrote results to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
