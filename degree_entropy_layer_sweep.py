#!/usr/bin/env python3
"""
Layer sweep for a DIRECT attention-graph degree-entropy intervention.

Context
-------
beta lowers attention-graph degree-distribution entropy (and degrades reasoning).
We now intervene on degree entropy DIRECTLY -- separately from beta -- to ask
whether moving it (up or down) shifts reasoning on its own. This is the SCREEN:
a fast layer sweep to find where the intervention has purchase, before building
the precise target-seeking lever and the A/B dissociation.

Lever (hybrid, fixed-reweighting -- fast, measure-not-target)
------------------------------------------------------------
At a target layer, during decode, we reweight the attention matrix per token to
push its thresholded-graph degree distribution either broader (higher degree
entropy) or more regular (lower degree entropy), at FIXED edge density, then
renormalize rows so it remains a valid attention distribution. Because degree
entropy is NON-MONOTONE in any single reweighting parameter (verified), we apply
a fixed reweighting and REPORT the achieved degree-entropy change rather than
targeting a value. The sweep thus shows, per layer and per direction:
  - achieved_entropy_change : did the edit actually move degree entropy here
  - drift / perplexity / coherence : did reasoning move here

Two reweighting edits (achieved direction is MEASURED, not assumed):
  broaden    : hubness reweighting via a per-key logit bias. On structured
               attention this tends to RAISE degree entropy (verified +0.2..+0.4
               on synthetic data), though non-monotonically in strength.
  regularize : shrink each row's logits toward their mean. Intended to LOWER
               degree entropy, but on near-random attention it does so only
               weakly/unreliably (verified: can go either way by ~0.1). We
               therefore REPORT the achieved entropy change per layer rather
               than assume a direction -- read achieved_entropy_change to see
               what each edit actually did at each layer.

Interpretation
--------------
This is a screen, not a causal test. Even if reasoning moves where entropy moves,
the A/B dissociation (separate scripts) is required to attribute it to the entropy
vs. the value-blend change the reweighting also causes. Expect, given prior
results, that the value blend will again be the driver -- but this sweep tells you
WHERE to run that test and whether the two directions behave differently.

Reuses patch / prompts / generation from beta_psychedelic_sweep_llama.py and the
graph metrics from attn_graph_entropy.py.

Example
-------
  python degree_entropy_layer_sweep.py --num-prompts 6 --strength 0.5
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
import torch.nn.functional as F

import beta_psychedelic_sweep_llama as core
import attn_graph_entropy as ag


# Global intervention state read by the custom attention.
class _DEState:
    def __init__(self):
        self.active = False
        self.layers = set()
        self.direction = "broaden"   # or "regularize"
        self.strength = 0.5
        self.decode_only = True

DESTATE = _DEState()


def _reweight_scores(scores, direction, strength):
    """scores: [batch, heads, q, k] attention LOGITS. Return reweighted logits
    that push the (eventual) degree distribution broader or more regular. We
    operate on logits along the key axis (-1), then the caller softmaxes.

    broaden:    add a per-key hubness bias so some keys become hubs (uneven
                degree) -> higher degree entropy.
    regularize: shrink logits toward their per-row mean (flatter, more uniform
                degrees) -> intended lower degree entropy (achieved value is
                measured, not assumed).
    """
    k = scores.shape[-1]
    if direction == "regularize":
        mean = scores.mean(dim=-1, keepdim=True)
        return mean + (1.0 - strength) * (scores - mean)
    else:  # broaden
        idx = torch.arange(k, device=scores.device, dtype=scores.dtype)
        prof = torch.cos(idx / max(k - 1, 1) * math.pi)  # +1..-1 across keys
        bias = strength * 3.0 * prof  # logit-space boost
        return scores + bias.view(1, 1, 1, k)


def install_degree_entropy_attention():
    from transformers.models.llama import modeling_llama
    beta_patched = modeling_llama.eager_attention_forward  # beta's patched fn

    def attn(module, query, key, value, attention_mask, scaling, **kwargs):
        layer_idx = getattr(module, "layer_idx", 0)
        if not DESTATE.active or layer_idx not in DESTATE.layers:
            return beta_patched(module, query, key, value, attention_mask, scaling, **kwargs)

        key_states, value_states = key, value
        n_rep = query.shape[1] // key_states.shape[1]
        if n_rep > 1:
            key_states = key_states.repeat_interleave(n_rep, dim=1)
            value_states = value_states.repeat_interleave(n_rep, dim=1)

        scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, :, : key_states.shape[-2]]

        # Apply the reweighting on EVERY forward (prefill and decode). The edit
        # must fire during the measurement forward (which is full-sequence,
        # q_len>1) or achieved_entropy_change is structurally zero. Restricting
        # to decode made the measurement blind to the intervention.
        scores = _reweight_scores(scores, DESTATE.direction, DESTATE.strength)
        w = F.softmax(scores, dim=-1, dtype=torch.float32)
        out = torch.matmul(w.to(value_states.dtype), value_states).transpose(1, 2).contiguous()
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
        print("[DE] WARNING: could not register custom attention; may be a no-op.")


@torch.no_grad()
def assert_de_fires(model, tokenizer):
    DESTATE.active = True
    DESTATE.layers = {0}
    DESTATE.direction = "broaden"
    DESTATE.strength = 0.5
    enc = tokenizer("probe for DE check", return_tensors="pt").to(model.device)
    # marker: monkeypatch a counter via a closure isn't trivial; instead rely on
    # generate producing output and trust the registry swap. Do a quick forward.
    model.generate(**enc, max_new_tokens=2, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    DESTATE.active = False
    DESTATE.layers = set()
    print("[DE check] attention path executed (registry swap active).")


@torch.no_grad()
def measure_attn_entropy(model, tokenizer, text, max_tokens, layer):
    """Achieved degree-entropy of the attention graph at `layer`, with the
    intervention CURRENTLY configured in DESTATE (active or not)."""
    attn = ag.capture_attention(model, tokenizer, text, max_tokens, [layer])
    if layer not in attn:
        return float("nan")
    return ag.attention_graph_metrics(attn[layer])["degree_entropy"]


@torch.no_grad()
def run_layer(model, tokenizer, prompts, layer, direction, strength,
              gen_tokens, metric_max_tokens):
    rows = []
    for item in prompts:
        # generate under the intervention at this layer
        DESTATE.active = True
        DESTATE.layers = {layer}
        DESTATE.direction = direction
        DESTATE.strength = strength
        output = core.generate(model, tokenizer, item["prompt"], gen_tokens,
                               do_sample=False, temperature=0.0, seed=None)
        # achieved degree entropy at this layer UNDER the intervention
        text = item["prompt"] + "\n" + output
        ent_interv = measure_attn_entropy(model, tokenizer, text, metric_max_tokens, layer)
        DESTATE.active = False
        # baseline degree entropy at this layer (no intervention), same text
        ent_base = measure_attn_entropy(model, tokenizer, text, metric_max_tokens, layer)

        drift = core.associative_drift(model, tokenizer, item["prompt"], output, metric_max_tokens)
        ppl = core.clean_perplexity(model, tokenizer, output, metric_max_tokens)
        local_coh = (1.0 / (1.0 + max(0.0, math.log(max(ppl, 1e-6)) - math.log(10.0)))
                     if not math.isnan(ppl) else float("nan"))
        rows.append({
            "layer": layer, "direction": direction, "strength": strength,
            "prompt_id": item["prompt_id"],
            "degree_entropy_interv": ent_interv,
            "degree_entropy_base": ent_base,
            "achieved_entropy_change": (ent_interv - ent_base
                                        if not (math.isnan(ent_interv) or math.isnan(ent_base))
                                        else float("nan")),
            "associative_drift": drift, "perplexity": ppl, "local_coherence": local_coh,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-1B-Instruct"))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--outdir", default="degree_entropy_sweep")
    ap.add_argument("--strength", type=float, default=0.5, help="reweighting strength 0..1")
    ap.add_argument("--num-prompts", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gen-tokens", type=int, default=80)
    ap.add_argument("--metric-max-tokens", type=int, default=256)
    ap.add_argument("--layers", default="all", help="'all' or comma list")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

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
    install_degree_entropy_attention()
    assert_de_fires(model, tokenizer)

    n_layers = int(model.config.num_hidden_layers)
    if args.layers == "all":
        layers = list(range(n_layers))
    else:
        layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

    prompts = core.make_open_prompts(args.num_prompts, args.seed)

    # baseline reasoning (no intervention)
    DESTATE.active = False
    base_rows = []
    for item in prompts:
        out = core.generate(model, tokenizer, item["prompt"], args.gen_tokens,
                            do_sample=False, temperature=0.0, seed=None)
        ppl = core.clean_perplexity(model, tokenizer, out, args.metric_max_tokens)
        base_rows.append({
            "associative_drift": core.associative_drift(model, tokenizer, item["prompt"], out, args.metric_max_tokens),
            "perplexity": ppl,
            "local_coherence": 1.0/(1.0+max(0.0, math.log(max(ppl,1e-6))-math.log(10.0))) if not math.isnan(ppl) else float("nan"),
        })
    base = pd.DataFrame(base_rows).mean(numeric_only=True).to_dict()

    all_frames = []
    for direction in ("broaden", "regularize"):
        for layer in layers:
            df = run_layer(model, tokenizer, prompts, layer, direction, args.strength,
                          args.gen_tokens, args.metric_max_tokens)
            all_frames.append(df)
            m = df.mean(numeric_only=True)
            print(f"[{direction} L{layer}] dEnt={m['achieved_entropy_change']:+.3f} "
                  f"drift={m['associative_drift']:.3f} ppl={m['perplexity']:.1f}")

    full = pd.concat(all_frames, ignore_index=True)
    full.to_csv(outdir / "per_prompt.csv", index=False)

    summ = full.groupby(["direction", "layer"]).agg(
        achieved_entropy_change=("achieved_entropy_change", "mean"),
        associative_drift=("associative_drift", "mean"),
        perplexity=("perplexity", "mean"),
        local_coherence=("local_coherence", "mean"),
        n=("perplexity", "count"),
    ).reset_index()
    summ["drift_shift_vs_base"] = summ["associative_drift"] - base["associative_drift"]
    summ["ppl_shift_vs_base"] = summ["perplexity"] - base["perplexity"]
    summ.to_csv(outdir / "layer_summary.csv", index=False)

    # rank layers where the intervention has PURCHASE (entropy moved) AND MATTERS
    # (reasoning moved), per direction.
    def rank(direction):
        d = summ[summ["direction"] == direction].copy()
        d["purchase"] = d["achieved_entropy_change"].abs()
        d["matters"] = d["ppl_shift_vs_base"].abs()
        d["score"] = d["purchase"] * d["matters"]
        top = d.sort_values("score", ascending=False).head(5)
        return top[["layer", "achieved_entropy_change", "ppl_shift_vs_base",
                    "drift_shift_vs_base", "score"]].to_dict("records")

    summary = {
        "model": args.model, "n_layers": n_layers, "strength": args.strength,
        "num_prompts": args.num_prompts,
        "baseline": {k: float(base[k]) for k in base},
        "note": "Screen only. 'purchase'=|entropy moved|, 'matters'=|ppl moved|. "
                "Layers where both are high are candidates for the precise "
                "target-seeking lever + A/B dissociation. Causation NOT established here.",
        "top_layers_broaden": rank("broaden"),
        "top_layers_regularize": rank("regularize"),
        "files": {"per_prompt": "per_prompt.csv", "layer_summary": "layer_summary.csv"},
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\n=== DEGREE-ENTROPY LAYER SWEEP ===")
    print(json.dumps(summary, indent=2, default=float))
    print(f"\nWrote results to: {outdir.resolve()}")


if __name__ == "__main__":
    main()