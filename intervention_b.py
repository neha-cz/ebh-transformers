#!/usr/bin/env python3
"""
Intervention B -- the value-pathway sufficiency test.

Question
--------
Lowering attention beta at layers 2,3 does TWO things at once:
  (a) flattens those layers' attention-weight entropy, and
  (b) rewrites their weighted value-sum (the attention output), which
      propagates downstream and loosens reasoning.
We want to know whether (b) ALONE is sufficient to loosen reasoning, with the
attention-weight entropy held at baseline. If yes -> entropy is a bystander,
the value pathway carries the effect.

Method
------
Pin attention exactly at baseline (NO beta scaling, so attention-weight entropy
is unchanged by construction), and instead perturb the attention OUTPUT directly
at the target layers, calibrated to match the output rel-L2 shift that beta=0.45
produced (~1.5 in the value-mixing probe). Then generate continuations and score
drift + clean-perplexity, and compare to (i) clean baseline and (ii) the beta
intervention.

We perturb the input to o_proj (== the concatenated-head weighted value-sum,
captured before the output projection) via a forward_pre_hook. This leaves the
softmax weights -- and thus their entropy -- untouched.

Two perturbation modes (set --mode):
  noise : add random-direction Gaussian noise scaled to a target rel-L2.
          Structureless; the honest control for "does ANY output shift of this
          size loosen reasoning?"
  vscale: scale the value-sum vector toward/away from itself (gain != 1).
          Has structure tied to the real values; tests whether the KIND of
          shift matters, not just its size.

Calibration
-----------
For each forward pass we measure the realized rel-L2 shift and adapt the
perturbation strength toward --target-shift so the comparison to beta is fair.
The realized shift is logged so you can confirm the match.

Interpretation truth table (vs the beta result on the same prompts/metrics):
  B loosens ~ as much as beta -> value pathway sufficient; entropy a bystander.
  B does not loosen            -> value shift of this size/kind is not enough;
                                  either entropy matters or the STRUCTURE of
                                  beta's shift matters (try the other --mode).

Reuses generation + metrics from beta_psychedelic_sweep_llama.py. Keep that file
alongside this one.

Example
-------
  python intervention_b.py --layers 2,3 --mode noise  --target-shift 1.5 --num-prompts 8
  python intervention_b.py --layers 2,3 --mode vscale --target-shift 1.5 --num-prompts 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import pandas as pd
import torch

import beta_psychedelic_sweep_llama as core


# -----------------------------
# Output-perturbation hook
# -----------------------------

class OutputPerturber:
    """Adds a calibrated perturbation to the input of o_proj at target layers
    during DECODE only (seq_len == 1 with a populated KV cache), so the trip
    affects freshly decoded tokens, matching how the beta patch was applied.

    Attention weights are never touched -> attention-weight entropy is unchanged.
    """

    def __init__(self, model, target_layers, mode="noise", target_shift=1.5,
                 seed=0, decode_only=True):
        self.model = model
        self.target = set(int(x) for x in target_layers)
        self.mode = mode
        self.target_shift = float(target_shift)
        self.decode_only = decode_only
        self.active = False
        self.handles = []
        self.realized_shifts = []  # list of rel-L2 shifts actually applied
        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(seed)
        self._register()

    def _layers(self):
        return self.model.model.layers

    def _is_decode(self, x):
        # x: [batch, seq, hidden]; decode step has seq == 1.
        return x.shape[1] == 1

    def _perturb(self, x):
        """Return a perturbed copy of x ([B, S, H]) with realized rel-L2 ~ target."""
        orig = x
        if self.mode == "noise":
            noise = torch.randn(x.shape, generator=self._gen, dtype=torch.float32).to(x.device, x.dtype)
            # scale noise so ||noise|| / ||x|| == target_shift (per-sample)
            x_norm = torch.linalg.vector_norm(x.float())
            n_norm = torch.linalg.vector_norm(noise.float()) + 1e-9
            scale = (self.target_shift * x_norm / n_norm)
            new = x + scale * noise
        elif self.mode == "vscale":
            # Move x along its own direction by a gain chosen so the change has
            # rel-L2 == target_shift: ||(g-1)x|| / ||x|| = |g-1| = target_shift.
            # Sign randomized so it's not always amplification.
            sign = 1.0 if torch.rand(1, generator=self._gen).item() > 0.5 else -1.0
            g = 1.0 + sign * self.target_shift
            new = x * g
        else:
            raise ValueError(f"unknown mode {self.mode}")
        # record realized shift
        num = torch.linalg.vector_norm((new - orig).float()).item()
        den = torch.linalg.vector_norm(orig.float()).item() + 1e-9
        self.realized_shifts.append(num / den)
        return new

    def _register(self):
        for idx, layer in enumerate(self._layers()):
            attn = layer.self_attn
            if not hasattr(attn, "o_proj"):
                raise RuntimeError(f"layer {idx} self_attn has no o_proj; adjust hook target.")

            def pre_hook(module, args, kwargs, _idx=idx):
                if not self.active or _idx not in self.target:
                    return None
                # locate the tensor argument (o_proj(attn_output))
                x = None
                pos = None
                if len(args) >= 1 and torch.is_tensor(args[0]):
                    x, pos = args[0], 0
                elif "input" in kwargs and torch.is_tensor(kwargs["input"]):
                    x = kwargs["input"]
                if x is None:
                    return None
                if self.decode_only and not self._is_decode(x):
                    return None
                new_x = self._perturb(x)
                if pos == 0:
                    new_args = (new_x,) + tuple(args[1:])
                    return (new_args, kwargs)
                else:
                    kwargs = dict(kwargs)
                    kwargs["input"] = new_x
                    return (args, kwargs)

            h = attn.o_proj.register_forward_pre_hook(pre_hook, with_kwargs=True)
            self.handles.append(h)

    def reset_log(self):
        self.realized_shifts = []

    def mean_realized_shift(self):
        return float(sum(self.realized_shifts) / len(self.realized_shifts)) if self.realized_shifts else float("nan")

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# -----------------------------
# Scoring one condition
# -----------------------------

@torch.no_grad()
def score_condition(model, tokenizer, prompts, *, perturber, beta_ratio, beta_layers,
                    gen_tokens, metric_max_tokens, base_seed, label):
    """Generate + score drift and clean-perplexity per prompt under whatever
    intervention is currently configured.

    Exactly one of these should be 'on' per call:
      - perturber.active = True (Intervention B), beta_ratio = 1.0
      - perturber.active = False, beta_ratio < 1.0 (the beta arm, for comparison)
      - both off (clean baseline)
    """
    rows = []
    for item in prompts:
        if perturber is not None:
            perturber.reset_log()
        # generation under the chosen intervention
        with core.beta_intervention(beta_ratio=beta_ratio, layers=beta_layers):
            output = core.generate(
                model, tokenizer, item["prompt"], gen_tokens,
                do_sample=False, temperature=0.0, seed=None,
            )
        drift = core.associative_drift(model, tokenizer, item["prompt"], output, metric_max_tokens)
        ppl = core.clean_perplexity(model, tokenizer, output, metric_max_tokens)
        local_coh = (1.0 / (1.0 + max(0.0, math.log(max(ppl, 1e-6)) - math.log(10.0)))
                     if not math.isnan(ppl) else float("nan"))
        health = core.output_health(output)
        rows.append({
            "condition": label,
            "prompt_id": item["prompt_id"],
            "output": output,
            "associative_drift": drift,
            "perplexity": ppl,
            "local_coherence": local_coh,
            "degenerate": health["degenerate"],
            "realized_output_shift": perturber.mean_realized_shift() if perturber and perturber.active else 0.0,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-1B-Instruct"))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--outdir", default="intervention_b")
    ap.add_argument("--layers", default="2,3", help="target layers")
    ap.add_argument("--mode", default="noise", choices=["noise", "vscale"])
    ap.add_argument("--target-shift", type=float, default=1.5,
                    help="output rel-L2 shift to match (beta=0.45 gave ~1.5)")
    ap.add_argument("--beta-compare", type=float, default=0.45,
                    help="beta_ratio for the comparison arm")
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gen-tokens", type=int, default=96)
    ap.add_argument("--metric-max-tokens", type=int, default=512)
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

    perturber = OutputPerturber(
        model, target_layers, mode=args.mode,
        target_shift=args.target_shift, seed=args.seed, decode_only=True,
    )

    frames = []

    # 1) clean baseline: no beta, no perturbation
    perturber.active = False
    frames.append(score_condition(
        model, tokenizer, prompts, perturber=perturber,
        beta_ratio=1.0, beta_layers=None,
        gen_tokens=args.gen_tokens, metric_max_tokens=args.metric_max_tokens,
        base_seed=args.seed, label="baseline",
    ))

    # 2) beta arm (for comparison): beta on target layers, no perturbation
    perturber.active = False
    frames.append(score_condition(
        model, tokenizer, prompts, perturber=perturber,
        beta_ratio=args.beta_compare, beta_layers=target_layers,
        gen_tokens=args.gen_tokens, metric_max_tokens=args.metric_max_tokens,
        base_seed=args.seed, label=f"beta_{args.beta_compare:g}",
    ))

    # 3) Intervention B: perturbation on, beta off (entropy pinned at baseline)
    perturber.active = True
    frames.append(score_condition(
        model, tokenizer, prompts, perturber=perturber,
        beta_ratio=1.0, beta_layers=None,
        gen_tokens=args.gen_tokens, metric_max_tokens=args.metric_max_tokens,
        base_seed=args.seed, label=f"interventionB_{args.mode}",
    ))
    perturber.active = False
    perturber.remove()

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(outdir / "per_prompt.csv", index=False)

    agg = df.groupby("condition", dropna=False).agg(
        associative_drift=("associative_drift", "mean"),
        associative_drift_sd=("associative_drift", "std"),
        local_coherence=("local_coherence", "mean"),
        perplexity=("perplexity", "mean"),
        degenerate=("degenerate", "mean"),
        realized_output_shift=("realized_output_shift", "mean"),
        n=("associative_drift", "count"),
    ).reset_index()
    agg.to_csv(outdir / "summary_by_condition.csv", index=False)

    # Verdict: did Intervention B reproduce beta's loosening?
    def row(label_prefix):
        m = agg[agg["condition"].str.startswith(label_prefix)]
        return m.iloc[0].to_dict() if not m.empty else None

    base = row("baseline")
    beta = row("beta_")
    bI = row("interventionB_")
    verdict = {}
    if base and beta and bI:
        beta_drift_gain = beta["associative_drift"] - base["associative_drift"]
        b_drift_gain = bI["associative_drift"] - base["associative_drift"]
        frac = (b_drift_gain / beta_drift_gain) if abs(beta_drift_gain) > 1e-6 else float("nan")
        verdict = {
            "baseline_drift": base["associative_drift"],
            "beta_drift": beta["associative_drift"],
            "interventionB_drift": bI["associative_drift"],
            "beta_drift_gain_over_baseline": beta_drift_gain,
            "interventionB_drift_gain_over_baseline": b_drift_gain,
            "fraction_of_beta_effect_reproduced": frac,
            "interventionB_realized_output_shift": bI["realized_output_shift"],
            "target_shift": args.target_shift,
            "interventionB_degenerate": bI["degenerate"],
            "value_pathway_sufficient": bool(
                not math.isnan(frac) and frac >= 0.5 and bI["degenerate"] < 0.5
            ),
            "interpretation": (
                "Intervention B reproduced >=50% of beta's drift increase with "
                "attention-weight entropy pinned at baseline -> the value pathway "
                "is sufficient; attention-entropy flattening is not necessary."
                if (not math.isnan(frac) and frac >= 0.5 and bI["degenerate"] < 0.5) else
                "Intervention B did NOT reproduce beta's loosening at matched output "
                "shift. Either entropy flattening matters, or the STRUCTURE of beta's "
                "output shift matters (try --mode vscale vs noise). Also check the "
                "realized_output_shift actually matched target, and that B is not "
                "merely producing degenerate text."
            ),
        }

    summary = {
        "model": args.model,
        "target_layers": target_layers,
        "mode": args.mode,
        "target_shift": args.target_shift,
        "beta_compare": args.beta_compare,
        "num_prompts": args.num_prompts,
        "note": "Intervention B pins attention weights (entropy unchanged) and "
                "perturbs the attention output directly to match beta's output shift.",
        "verdict": verdict,
        "files": {"per_prompt": "per_prompt.csv", "summary_by_condition": "summary_by_condition.csv"},
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n=== INTERVENTION B SUMMARY ===")
    print(json.dumps(summary, indent=2, default=float))
    print(f"\nWrote results to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
