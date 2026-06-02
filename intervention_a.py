#!/usr/bin/env python3
"""
Intervention A -- the entropy-necessity test (clean attention-level version).

Question
--------
Does raising attention-weight ENTROPY (flattening the weights toward uniform)
loosen reasoning, when the attention OUTPUT is held close to baseline? Necessity
counterpart to Intervention B (sufficiency of the value pathway).

How it works
------------
Installs its OWN eager-attention replacement (same registry the beta patch uses),
where Q, K, V, and scaling are directly available. On the target layers, during
decode:

    scores   = Q @ K^T * scaling
    w        = softmax(scores + mask)              (baseline weights)
    w_flat   = (1-flatten)*w + flatten*uniform     (entropy raised)
    out_base = w      @ V
    out_flat = w_flat @ V
    out_final = out_flat + restore*(out_base - out_flat)

restore=1.0 cancels the output change entirely, so the returned output equals
baseline EXACTLY -- only the (internal) weight entropy was raised. The model's
downstream sees the same output, so behavior cannot change. That is the point:
it demonstrates that raising weight entropy *without* changing the output does
nothing, because only the output propagates. The informative regime is
restore < 1.0, where partial restoration lets you trace how much of beta's
effect tracks the entropy-induced output change.

INTERPRETATION (asymmetric):
  restore=1.0, entropy raised, NO behavioral change
     -> trivially expected (output identical); confirms entropy-without-output
        does nothing. Use as a sanity check.
  restore<1.0, entropy raised, output partly changed:
     compare the behavioral change to the residual output shift. If a large
     entropy increase with a SMALL residual output shift produces little
     behavioral change, entropy is NOT necessary (robust: residual biases
     toward an effect). If behavior changes mainly when the residual is large,
     it's the output change, not the entropy.

Reuses generation + metrics from beta_psychedelic_sweep_llama.py.

Example
-------
  python intervention_a.py --layers 2,3 --flatten 0.6 --restore 1.0 --num-prompts 8
  python intervention_a.py --layers 2,3 --flatten 0.6 --restore 0.5 --num-prompts 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

import beta_psychedelic_sweep_llama as core


class _AState:
    def __init__(self):
        self.active = False
        self.layers = set()
        self.flatten = 0.6
        self.restore = 1.0
        self.decode_only = True
        self.entropy_increases = []
        self.output_shifts = []

    def reset_log(self):
        self.entropy_increases = []
        self.output_shifts = []

ASTATE = _AState()


def install_entropy_attention():
    """Replace eager attention with a version that flattens weights + restores
    output on ASTATE.layers, delegating to the captured original elsewhere."""
    from transformers.models.llama import modeling_llama
    # Delegate to the CURRENT (beta-patched) attention, NOT the original, so the
    # beta arm still applies beta when A is inactive/off-target. Capturing
    # _beta_sweep_original here was a bug: it bypassed beta entirely.
    beta_patched = modeling_llama.eager_attention_forward  # this IS core's patched fn

    def attn(module, query, key, value, attention_mask, scaling, **kwargs):
        layer_idx = getattr(module, "layer_idx", 0)
        if not ASTATE.active or layer_idx not in ASTATE.layers:
            # Hand off to beta's patched attention so beta_intervention state is honored.
            return beta_patched(module, query, key, value, attention_mask, scaling, **kwargs)

        # On A's target layers we run our own path. Honor any active beta ratio
        # on these layers too, by folding it into scaling exactly as core does.
        try:
            beta_r = core.INTERVENTION.ratio_for_layer(layer_idx)
        except Exception:
            beta_r = 1.0
        scaling = scaling * beta_r

        key_states, value_states = key, value
        n_rep = query.shape[1] // key_states.shape[1]
        if n_rep > 1:
            key_states = key_states.repeat_interleave(n_rep, dim=1)
            value_states = value_states.repeat_interleave(n_rep, dim=1)

        scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, :, : key_states.shape[-2]]
        w = F.softmax(scores, dim=-1, dtype=torch.float32)

        q_len = w.shape[2]
        if ASTATE.decode_only and q_len != 1:
            out = torch.matmul(w.to(value_states.dtype), value_states)
            out = out.transpose(1, 2).contiguous()
            return out, w

        k = w.shape[-1]
        uniform = torch.full_like(w, 1.0 / k)
        w_flat = (1.0 - ASTATE.flatten) * w + ASTATE.flatten * uniform

        def nent(p):
            pp = p.clamp_min(1e-12)
            e = -(pp * pp.log()).sum(dim=-1)
            return (e / math.log(max(k, 2))).mean().item()
        ASTATE.entropy_increases.append(nent(w_flat) - nent(w))

        Vf = value_states.float()
        out_base = torch.matmul(w, Vf)
        out_flat = torch.matmul(w_flat, Vf)
        out_final = out_flat + ASTATE.restore * (out_base - out_flat)

        num = torch.linalg.vector_norm(out_final - out_base).item()
        den = torch.linalg.vector_norm(out_base).item() + 1e-9
        ASTATE.output_shifts.append(num / den)

        out = out_final.to(value_states.dtype).transpose(1, 2).contiguous()
        return out, w

    modeling_llama.eager_attention_forward = attn
    installed = False
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS["eager"] = attn
        installed = True
    except Exception:
        pass
    if not installed:
        try:
            from transformers.modeling_utils import AttentionInterface
            if hasattr(AttentionInterface, "_global_mapping"):
                AttentionInterface._global_mapping["eager"] = attn
                installed = True
            elif hasattr(AttentionInterface, "register"):
                AttentionInterface.register("eager", attn)
                installed = True
        except Exception:
            pass
    if not installed:
        print("[A] WARNING: could not register custom attention; A may be a no-op.")
    return installed


@torch.no_grad()
def assert_a_fires(model, tokenizer):
    ASTATE.active = True
    ASTATE.layers = {0}
    ASTATE.reset_log()
    enc = tokenizer("probe sentence for A check", return_tensors="pt").to(model.device)
    model.generate(**enc, max_new_tokens=2, do_sample=False,
                   pad_token_id=tokenizer.pad_token_id)
    fired = len(ASTATE.entropy_increases) > 0
    ASTATE.active = False
    ASTATE.layers = set()
    if not fired:
        raise RuntimeError("Intervention A attention did not fire on decode; "
                           "custom attention is not on the active path.")
    print(f"[A check] OK, fired {len(ASTATE.entropy_increases)} times.")


@torch.no_grad()
def score_condition(model, tokenizer, prompts, *, a_on, beta_ratio, beta_layers,
                    gen_tokens, metric_max_tokens, label):
    rows = []
    for item in prompts:
        ASTATE.reset_log()
        ASTATE.active = a_on
        with core.beta_intervention(beta_ratio=beta_ratio, layers=beta_layers):
            output = core.generate(model, tokenizer, item["prompt"], gen_tokens,
                                    do_sample=False, temperature=0.0, seed=None)
        ASTATE.active = False
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
            "realized_entropy_increase": (float(sum(ASTATE.entropy_increases) / len(ASTATE.entropy_increases))
                                          if a_on and ASTATE.entropy_increases else 0.0),
            "realized_output_shift": (float(sum(ASTATE.output_shifts) / len(ASTATE.output_shifts))
                                      if a_on and ASTATE.output_shifts else 0.0),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-1B-Instruct"))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--outdir", default="intervention_a")
    ap.add_argument("--layers", default="2,3")
    ap.add_argument("--flatten", type=float, default=0.6, help="toward uniform, 0..1")
    ap.add_argument("--restore", type=float, default=1.0, help="cancel output change, 0..1")
    ap.add_argument("--beta-compare", type=float, default=0.45)
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
    install_entropy_attention()
    assert_a_fires(model, tokenizer)

    ASTATE.layers = set(target_layers)
    ASTATE.flatten = args.flatten
    ASTATE.restore = args.restore

    prompts = core.make_open_prompts(args.num_prompts, args.seed)

    frames = []
    frames.append(score_condition(model, tokenizer, prompts, a_on=False,
                                   beta_ratio=1.0, beta_layers=None,
                                   gen_tokens=args.gen_tokens,
                                   metric_max_tokens=args.metric_max_tokens,
                                   label="baseline"))
    frames.append(score_condition(model, tokenizer, prompts, a_on=False,
                                   beta_ratio=args.beta_compare, beta_layers=target_layers,
                                   gen_tokens=args.gen_tokens,
                                   metric_max_tokens=args.metric_max_tokens,
                                   label=f"beta_{args.beta_compare:g}"))
    frames.append(score_condition(model, tokenizer, prompts, a_on=True,
                                   beta_ratio=1.0, beta_layers=None,
                                   gen_tokens=args.gen_tokens,
                                   metric_max_tokens=args.metric_max_tokens,
                                   label="interventionA"))

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(outdir / "per_prompt.csv", index=False)
    agg = df.groupby("condition", dropna=False).agg(
        associative_drift=("associative_drift", "mean"),
        local_coherence=("local_coherence", "mean"),
        perplexity=("perplexity", "mean"),
        degenerate=("degenerate", "mean"),
        realized_entropy_increase=("realized_entropy_increase", "mean"),
        realized_output_shift=("realized_output_shift", "mean"),
        n=("associative_drift", "count"),
    ).reset_index()
    agg.to_csv(outdir / "summary_by_condition.csv", index=False)

    def row(pfx):
        m = agg[agg["condition"].str.startswith(pfx)]
        return m.iloc[0].to_dict() if not m.empty else None
    base, beta, a = row("baseline"), row("beta_"), row("interventionA")

    verdict = {}
    if base and beta and a:
        # Sanity: if the beta arm equals baseline, beta did NOT apply (patch
        # collision) and any fraction-of-beta comparison is meaningless.
        beta_moved = (abs(beta["perplexity"] - base["perplexity"]) > 1e-6
                      or abs(beta["associative_drift"] - base["associative_drift"]) > 1e-6)
        if not beta_moved:
            verdict = {
                "ERROR": "beta arm is identical to baseline -> beta did not apply "
                         "this run (patch collision). Fix delegation before trusting "
                         "any A verdict; the fraction-reproduced comparison is invalid.",
                "realized_entropy_increase": a["realized_entropy_increase"],
                "realized_output_shift_residual": a["realized_output_shift"],
            }
            summary_verdict_ok = False
        else:
            summary_verdict_ok = True

        def frac(x):
            g = beta[x] - base[x]
            return (a[x] - base[x]) / g if abs(g) > 1e-6 else float("nan")
        ent_inc = a["realized_entropy_increase"]
        residual = a["realized_output_shift"]
        coh_frac = frac("perplexity")
        if summary_verdict_ok:
            verdict = {
                "realized_entropy_increase": ent_inc,
                "realized_output_shift_residual": residual,
                "restore": args.restore,
                "perplexity_fraction_reproduced": coh_frac,
                "drift_fraction_reproduced": frac("associative_drift"),
                "interventionA_degenerate": a["degenerate"],
            }
            big_entropy = ent_inc > 0.1
            reproduced = (not math.isnan(coh_frac)) and coh_frac >= 0.3
            small_residual = residual < 0.2
            if args.restore >= 0.999 and big_entropy:
                verdict["entropy_necessary"] = bool(reproduced)
                verdict["interpretation"] = (
                    "restore=1.0: returned output equals baseline exactly, so any "
                    "reproduced effect comes from the weight-entropy change alone. " + (
                        "An effect WAS seen -> entropy change alone moves behavior."
                        if reproduced else
                        "No effect -> raising weight entropy with output held fixed does "
                        "NOTHING; entropy alone is not driving behavior."
                    )
                )
            elif big_entropy and not reproduced:
                verdict["entropy_necessary"] = False
                verdict["interpretation"] = (
                    "Entropy raised substantially, coherence effect not reproduced -> "
                    "entropy flattening is NOT necessary for beta's effect (robust: "
                    "residual output change biases toward an effect).")
            elif big_entropy and reproduced and small_residual:
                verdict["entropy_necessary"] = True
                verdict["interpretation"] = (
                    "Entropy raised, output well-restored, effect reproduced -> entropy "
                    "appears to drive the effect.")
            elif big_entropy and reproduced and not small_residual:
                verdict["entropy_necessary"] = None
                verdict["interpretation"] = (
                    f"Effect reproduced but residual output shift large ({residual:.2f}) "
                    "-> AMBIGUOUS; raise --restore toward 1.0.")
            else:
                verdict["entropy_necessary"] = None
                verdict["interpretation"] = (
                    f"Entropy not raised enough (increase={ent_inc:.3f}); raise --flatten.")

    summary = {
        "model": args.model, "target_layers": target_layers,
        "flatten": args.flatten, "restore": args.restore,
        "beta_compare": args.beta_compare, "num_prompts": args.num_prompts,
        "verdict": verdict,
        "files": {"per_prompt": "per_prompt.csv", "summary_by_condition": "summary_by_condition.csv"},
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\n=== INTERVENTION A SUMMARY ===")
    print(json.dumps(summary, indent=2, default=float))
    print(f"\nWrote results to: {outdir.resolve()}")


if __name__ == "__main__":
    main()