#!/usr/bin/env python3
"""
Value-mixing probe: does beta-scaling change the attention OUTPUT (the weighted
value-sum written to the residual stream) much more than it changes the ENTROPY
of the attention weights?

Motivation
----------
The layer sweep showed per-layer attention entropy barely moves even where
coherence collapses (layer 3: perplexity 3.5 -> 26, entropy change ~ -0.003).
That rules out "blurred attention distribution" as the mechanism. The leading
alternative is value-mixing: beta perturbs the logits -> shifts which value
vectors get blended (output = sum_i w_i * v_i) -> changes downstream
representations, WITHOUT changing the entropy of w. Entropy is a property of w
alone; the output depends on w AND v, so they can move independently.

This probe measures both, per layer, baseline vs an intervention beta:
  - attention OUTPUT shift: relative L2 change in the per-token weighted
    value-sum (captured as the input to o_proj, i.e. concatenated heads before
    the output projection). LARGE = value-mixing is happening.
  - attention WEIGHT entropy shift: change in normalized attention entropy.
    SMALL = the weight distribution is NOT what's changing.
  - pre-softmax logit scale: std of QK logits per layer, to show what softmax
    regime each layer sits in (explains why entropy is insensitive to scaling).

If output_shift is large while entropy_shift is small at the layers that drive
coherence loss (2, 3), value-mixing is demonstrated, not just hypothesized.

Reuses the patch / prompts / generation from beta_psychedelic_sweep_llama.py.
Keep that file in the same directory.

Example
-------
    python value_mixing_probe.py --beta 0.4 --layers 2,3 --num-prompts 8
    python value_mixing_probe.py --beta 0.35 --layers all --outdir vmprobe
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
# Capture hooks
# -----------------------------

class AttnCapture:
    """Captures, per decoder layer:
      - the input to o_proj  (== concatenated-head weighted value-sum, the
        attention OUTPUT before output projection): tensor [batch, seq, hidden]
      - normalized attention-weight entropy (mean over heads/queries)
      - pre-softmax QK logit std (a proxy for softmax regime)

    o_proj input is the most stable cross-version handle on sum_i w_i v_i: it is
    exactly the per-token blended value, concatenated across heads, right before
    the output projection. We grab it with a forward_pre_hook on o_proj.
    """

    def __init__(self, model):
        self.model = model
        self.o_proj_input = {}      # layer_idx -> tensor [B, S, H]
        self.handles = []
        self._register()

    def _layers(self):
        # transformers exposes decoder layers at model.model.layers
        return self.model.model.layers

    def _register(self):
        for idx, layer in enumerate(self._layers()):
            attn = layer.self_attn
            if not hasattr(attn, "o_proj"):
                raise RuntimeError(
                    f"layer {idx} self_attn has no o_proj; this transformers "
                    "version names the output projection differently. Inspect "
                    "LlamaAttention and adjust the hook target."
                )

            def pre_hook(module, args, kwargs, _idx=idx):
                # o_proj is called as o_proj(attn_output). Capture its input.
                x = None
                if len(args) >= 1 and torch.is_tensor(args[0]):
                    x = args[0]
                elif "input" in kwargs and torch.is_tensor(kwargs["input"]):
                    x = kwargs["input"]
                if x is not None:
                    self.o_proj_input[_idx] = x.detach().float().cpu()
                return None

            # with_kwargs=True so we can read kwargs too (version-dependent call style)
            h = attn.o_proj.register_forward_pre_hook(pre_hook, with_kwargs=True)
            self.handles.append(h)

    def clear(self):
        self.o_proj_input = {}

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def forward_capture(model, tokenizer, text, cap: AttnCapture, max_tokens):
    """Run a clean forward (also requesting attentions for entropy + logit std
    via the attention maps) and return (o_proj_inputs, entropy_by_layer,
    logit_std_by_layer)."""
    cap.clear()
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens).to(model.device)
    out = model(**enc, output_attentions=True, use_cache=False)

    # entropy per layer from attention maps
    ent_by_layer = {}
    logit_std_by_layer = {}
    eps = 1e-12
    if out.attentions is not None:
        for li, attn in enumerate(out.attentions):
            p = attn.float().clamp_min(eps)
            ent = -(p * p.log()).sum(dim=-1)
            key_len = p.shape[-1]
            norm = math.log(max(key_len, 2))
            ent_by_layer[li] = (ent / norm).mean().item()
            # logit-std proxy: recover from probabilities is lossy, so instead
            # approximate the "peakedness" via max prob; std of logits is not
            # directly available post-softmax. We report mean max-prob as the
            # regime indicator (high max-prob = peaked/saturated regime).
            logit_std_by_layer[li] = p.max(dim=-1).values.mean().item()  # mean max attention prob

    outputs = {li: t.clone() for li, t in cap.o_proj_input.items()}
    return outputs, ent_by_layer, logit_std_by_layer


def rel_l2_shift(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 change between two [B,S,H] tensors: ||b-a|| / ||a||,
    averaged appropriately by flattening."""
    if a is None or b is None:
        return float("nan")
    if a.shape != b.shape:
        n = min(a.shape[1], b.shape[1])
        a = a[:, :n, :]
        b = b[:, :n, :]
    num = torch.linalg.vector_norm(b - a).item()
    den = torch.linalg.vector_norm(a).item() + 1e-9
    return float(num / den)


def cosine_shift(a: torch.Tensor, b: torch.Tensor) -> float:
    """1 - mean per-token cosine similarity between the two output tensors.
    Complements rel_l2 (scale-invariant view of how much direction changed)."""
    if a is None or b is None:
        return float("nan")
    if a.shape != b.shape:
        n = min(a.shape[1], b.shape[1])
        a = a[:, :n, :]
        b = b[:, :n, :]
    av = a.reshape(-1, a.shape[-1])
    bv = b.reshape(-1, b.shape[-1])
    av = torch.nn.functional.normalize(av, dim=-1)
    bv = torch.nn.functional.normalize(bv, dim=-1)
    cos = (av * bv).sum(-1)
    return float(1.0 - cos.mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-1B-Instruct"))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--outdir", default="value_mixing_probe")
    ap.add_argument("--beta", type=float, default=0.4, help="intervention beta_ratio")
    ap.add_argument("--layers", default="2,3",
                    help="layers to flatten under the intervention: comma list or 'all'")
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--metric-max-tokens", type=int, default=512)
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

    n_layers = int(model.config.num_hidden_layers)
    if args.layers == "all":
        flatten_layers = list(range(n_layers))
    else:
        flatten_layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

    prompts = core.make_open_prompts(args.num_prompts, args.seed)
    cap = AttnCapture(model)

    per_prompt_rows = []
    for item in prompts:
        text = item["prompt"]

        # Baseline forward (no intervention)
        with core.beta_intervention(beta_ratio=1.0, layers=None):
            base_out, base_ent, base_reg = forward_capture(
                model, tokenizer, text, cap, args.metric_max_tokens)

        # Intervened forward
        with core.beta_intervention(beta_ratio=args.beta, layers=flatten_layers):
            int_out, int_ent, int_reg = forward_capture(
                model, tokenizer, text, cap, args.metric_max_tokens)

        for li in range(n_layers):
            a = base_out.get(li)
            b = int_out.get(li)
            per_prompt_rows.append({
                "prompt_id": item["prompt_id"],
                "layer": li,
                "flattened": li in flatten_layers,
                "output_rel_l2_shift": rel_l2_shift(a, b),
                "output_cosine_shift": cosine_shift(a, b),
                "entropy_baseline": base_ent.get(li, float("nan")),
                "entropy_intervened": int_ent.get(li, float("nan")),
                "entropy_shift": (int_ent.get(li, float("nan")) - base_ent.get(li, float("nan")))
                                 if (li in int_ent and li in base_ent) else float("nan"),
                "mean_maxprob_baseline": base_reg.get(li, float("nan")),
                "mean_maxprob_intervened": int_reg.get(li, float("nan")),
            })

    cap.remove()
    df = pd.DataFrame(per_prompt_rows)
    df.to_csv(outdir / "per_prompt_layer_shifts.csv", index=False)

    # Aggregate per layer
    agg = df.groupby(["layer", "flattened"], dropna=False).agg(
        output_rel_l2_shift=("output_rel_l2_shift", "mean"),
        output_cosine_shift=("output_cosine_shift", "mean"),
        entropy_shift=("entropy_shift", "mean"),
        mean_maxprob_baseline=("mean_maxprob_baseline", "mean"),
        n=("output_rel_l2_shift", "count"),
    ).reset_index().sort_values("layer")
    agg.to_csv(outdir / "layer_shift_summary.csv", index=False)

    # Verdict on the flattened layers: is output shift >> entropy shift?
    flat = agg[agg["flattened"]]
    verdict = {}
    if not flat.empty:
        mean_out = float(flat["output_rel_l2_shift"].mean())
        mean_ent = float(flat["entropy_shift"].abs().mean())
        # ratio: how many times larger is the output shift than the entropy shift
        ratio = mean_out / (mean_ent + 1e-9)
        verdict = {
            "flattened_layers": flatten_layers,
            "mean_output_rel_l2_shift": mean_out,
            "mean_abs_entropy_shift": mean_ent,
            "output_to_entropy_ratio": ratio,
            "value_mixing_supported": bool(mean_out > 0.05 and ratio > 5.0),
            "interpretation": (
                "value-mixing demonstrated: beta moves the attention OUTPUT "
                "(blended value vectors) substantially while barely changing "
                "weight entropy -> the effect rides on which values get mixed, "
                "not on attention blurring"
                if (mean_out > 0.05 and ratio > 5.0) else
                "output shift is not large relative to entropy shift here; "
                "value-mixing is NOT clearly the pathway -- the drift may originate "
                "even further downstream, or thresholds need revisiting"
            ),
        }

    # Plots
    try:
        import matplotlib.pyplot as plt
        x = agg["layer"].astype(int)
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.bar(x - 0.2, agg["output_rel_l2_shift"], width=0.4,
                label="attention output rel-L2 shift", color="#C44E52")
        ax1.set_xlabel("layer")
        ax1.set_ylabel("output rel-L2 shift", color="#C44E52")
        ax2 = ax1.twinx()
        ax2.bar(x + 0.2, agg["entropy_shift"], width=0.4,
                label="attention entropy shift", color="#4C72B0")
        ax2.set_ylabel("entropy shift", color="#4C72B0")
        plt.title(f"Output shift vs entropy shift per layer (beta={args.beta}, "
                  f"flattened={args.layers})")
        fig.tight_layout()
        fig.savefig(outdir / "output_vs_entropy_shift.png", dpi=160)
        plt.close(fig)
    except Exception as e:
        print(f"[plot skipped] {e}")

    summary = {
        "model": args.model,
        "n_layers": n_layers,
        "beta": args.beta,
        "flattened_layers": flatten_layers,
        "num_prompts": args.num_prompts,
        "patch_calls": core.PATCH_CALLS,
        "verdict": verdict,
        "metric_notes": {
            "output_rel_l2_shift": "||intervened - baseline|| / ||baseline|| of the "
                                   "pre-o_proj weighted value-sum; LARGE = value-mixing",
            "entropy_shift": "change in normalized attention-weight entropy; SMALL = "
                             "weight distribution is not what's moving",
            "mean_maxprob_baseline": "mean max attention prob; high = peaked/saturated "
                                     "softmax regime (explains entropy insensitivity)",
        },
        "files": {
            "per_prompt_layer_shifts": "per_prompt_layer_shifts.csv",
            "layer_shift_summary": "layer_shift_summary.csv",
            "plot": "output_vs_entropy_shift.png",
        },
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n=== VALUE-MIXING PROBE SUMMARY ===")
    print(json.dumps(summary, indent=2, default=float))
    print(f"\nWrote results to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
