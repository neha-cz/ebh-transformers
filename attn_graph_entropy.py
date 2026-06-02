#!/usr/bin/env python3
"""
Attention-graph degree-distribution Shannon entropy: baseline vs beta.

Motivation
----------
Viol et al. (2017, Sci Rep) measured the Shannon entropy of the DEGREE
DISTRIBUTION of brain functional-connectivity networks and found it INCREASES
under ayahuasca, alongside increased local integration and decreased global
integration. This is a different quantity from attention-weight entropy or
spatial entropy: it is the entropy of the *connectivity graph's* degree
distribution -- how uneven the graph's hub structure is.

The transformer-native analog: attention IS a token-token connectivity graph.
Threshold the attention weights into edges, take each token's degree, and
compute the Shannon entropy of the degree distribution across tokens. This is
both a faithful analog of the paper's measure AND (unlike LZc / spatial entropy)
directly interveneable, because it is a function of the attention matrix we
already know how to patch. This script is the SCREENING step: does the measure
even move under beta, before we build an intervention on it.

Method (mirrors the paper's methodology)
----------------------------------------
- Build one undirected graph per layer per prompt from the attention matrix:
  symmetrize (A + A^T)/2, average over heads, drop the causal-mask zeros.
- CRITICAL: the paper compares at MATCHED EDGE DENSITY, not matched threshold,
  because degree-entropy depends on density. We therefore threshold each graph
  to a target density (keep the top-x% strongest edges) and compare baseline vs
  beta at the SAME density, sweeping several densities.
- Compute, per graph: Shannon entropy of the degree distribution, mean
  clustering coefficient (local integration), and global efficiency (global
  integration) -- the paper's three companion quantities.

Caveats baked into interpretation
----------------------------------
- Attention is directed/per-pass/task-driven; brain FC is undirected/resting.
  We symmetrize to approximate their undirected construction, but the semantics
  differ -- state this.
- Screening only: shows whether the measure moves under beta, NOT whether it
  causally drives reasoning. That needs an intervention on degree entropy
  directly (the interveneable advantage of this measure over LZc/spatial).
- Threshold/density dependence is real; we sweep densities and compare matched.

Reuses patch / prompts / generation from beta_psychedelic_sweep_llama.py.

Example
-------
  python attn_graph_entropy.py --beta 0.45 --layers 2,3 --num-prompts 8
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

# Edge densities to compare at (fraction of possible edges kept). Matched across
# conditions, mirroring the paper's matched-mean-degree comparison.
DENSITIES = (0.05, 0.10, 0.15, 0.20, 0.30)


# -----------------------------
# Graph measures
# -----------------------------

def _degree_distribution_entropy(adj: np.ndarray) -> float:
    """Shannon entropy of the degree distribution of an undirected unweighted
    graph (adjacency matrix adj, 0/1, symmetric, zero diagonal). Natural log,
    matching the paper."""
    deg = adj.sum(axis=1).astype(int)
    n = len(deg)
    if n < 2:
        return float("nan")
    # distribution over observed degree values
    counts = np.bincount(deg)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def _clustering_coefficient(adj: np.ndarray) -> float:
    """Mean local clustering coefficient (local integration)."""
    n = adj.shape[0]
    if n < 3:
        return float("nan")
    cc = []
    for i in range(n):
        nbrs = np.where(adj[i] > 0)[0]
        kd = len(nbrs)
        if kd < 2:
            cc.append(0.0)
            continue
        sub = adj[np.ix_(nbrs, nbrs)]
        links = sub.sum() / 2.0
        cc.append(2.0 * links / (kd * (kd - 1)))
    return float(np.mean(cc)) if cc else float("nan")


def _global_efficiency(adj: np.ndarray) -> float:
    """Global efficiency = mean of 1/shortest_path over all node pairs (global
    integration). BFS shortest paths on the unweighted graph."""
    n = adj.shape[0]
    if n < 2:
        return float("nan")
    total = 0.0
    cnt = 0
    adj_bool = adj > 0
    for src in range(n):
        # BFS
        dist = -np.ones(n, dtype=int)
        dist[src] = 0
        frontier = [src]
        while frontier:
            nxt = []
            for u in frontier:
                for v in np.where(adj_bool[u])[0]:
                    if dist[v] < 0:
                        dist[v] = dist[u] + 1
                        nxt.append(v)
            frontier = nxt
        for v in range(n):
            if v != src and dist[v] > 0:
                total += 1.0 / dist[v]
                cnt += 1
    return float(total / (n * (n - 1))) if n > 1 else float("nan")


def _threshold_to_density(W: np.ndarray, density: float) -> np.ndarray:
    """Keep the top `density` fraction of off-diagonal edge weights as edges."""
    n = W.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = W[iu]
    if len(vals) == 0:
        return np.zeros_like(W)
    n_keep = max(1, int(round(density * len(vals))))
    if n_keep >= len(vals):
        thr = -np.inf
    else:
        thr = np.partition(vals, len(vals) - n_keep)[len(vals) - n_keep]
    adj = np.zeros((n, n), dtype=np.uint8)
    mask = W >= thr
    np.fill_diagonal(mask, False)
    adj[mask] = 1
    adj = np.maximum(adj, adj.T)  # ensure symmetric
    return adj


def attention_graph_metrics(attn_layer: np.ndarray, densities=DENSITIES) -> dict:
    """attn_layer: [heads, q, k] attention weights for one layer (single batch).
    Build an undirected weighted graph (avg heads, symmetrize), then at each
    target density compute degree-entropy, clustering, global efficiency.
    Returns dict of {metric: mean over densities}."""
    # average over heads -> [q, k]; q==k for full self-attention over the window
    W = attn_layer.mean(axis=0)
    n = min(W.shape)
    W = W[:n, :n]
    # symmetrize to undirected
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    if n < 4:
        return {"degree_entropy": float("nan"), "clustering": float("nan"),
                "global_efficiency": float("nan")}
    ent, clu, eff = [], [], []
    for d in densities:
        adj = _threshold_to_density(W, d)
        ent.append(_degree_distribution_entropy(adj))
        clu.append(_clustering_coefficient(adj))
        eff.append(_global_efficiency(adj))
    nanmean = lambda xs: float(np.nanmean(xs)) if len(xs) else float("nan")
    return {"degree_entropy": nanmean(ent), "clustering": nanmean(clu),
            "global_efficiency": nanmean(eff)}


# -----------------------------
# Capture attention at target layers (cheap hook, no output_attentions)
# -----------------------------

@torch.no_grad()
def capture_attention(model, tokenizer, text, max_tokens, layers):
    """Return {layer_idx: [heads, q, k] numpy} for the requested layers, via a
    lightweight hook (no output_attentions over all layers)."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens).to(model.device)
    grabbed = {}
    handles = []
    want = set(int(x) for x in layers)

    def make_hook(idx):
        def hook(module, args, kwargs, output):
            if isinstance(output, tuple) and len(output) >= 2 and torch.is_tensor(output[1]):
                grabbed[idx] = output[1][0].float().cpu().numpy()  # [heads, q, k]
            return output
        return hook

    for idx, layer in enumerate(model.model.layers):
        if idx in want:
            handles.append(layer.self_attn.register_forward_hook(make_hook(idx), with_kwargs=True))
    model(**enc, use_cache=False)
    for h in handles:
        h.remove()
    return grabbed


@torch.no_grad()
def run_condition(model, tokenizer, prompts, *, beta_ratio, beta_layers, target_layers,
                  gen_tokens, metric_max_tokens, label):
    rows = []
    for item in prompts:
        with core.beta_intervention(beta_ratio=beta_ratio, layers=beta_layers):
            output = core.generate(model, tokenizer, item["prompt"], gen_tokens,
                                    do_sample=False, temperature=0.0, seed=None)
            text = item["prompt"] + "\n" + output
            attn = capture_attention(model, tokenizer, text, metric_max_tokens, target_layers)

        drift = core.associative_drift(model, tokenizer, item["prompt"], output, metric_max_tokens)
        ppl = core.clean_perplexity(model, tokenizer, output, metric_max_tokens)
        local_coh = (1.0 / (1.0 + max(0.0, math.log(max(ppl, 1e-6)) - math.log(10.0)))
                     if not math.isnan(ppl) else float("nan"))

        # average graph metrics over the target layers
        per_layer = [attention_graph_metrics(attn[l]) for l in target_layers if l in attn]
        def avg(key):
            vals = [m[key] for m in per_layer if not math.isnan(m[key])]
            return float(np.mean(vals)) if vals else float("nan")

        rows.append({
            "condition": label,
            "prompt_id": item["prompt_id"],
            "associative_drift": drift,
            "perplexity": ppl,
            "local_coherence": local_coh,
            "degree_entropy": avg("degree_entropy"),
            "clustering": avg("clustering"),
            "global_efficiency": avg("global_efficiency"),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-1B-Instruct"))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--outdir", default="attn_graph_entropy")
    ap.add_argument("--beta", type=float, default=0.45)
    ap.add_argument("--layers", default="2,3")
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gen-tokens", type=int, default=96)
    ap.add_argument("--metric-max-tokens", type=int, default=256,
                    help="cap tokens; graph metrics are O(n^2-n^3) in token count")
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

    frames = []
    frames.append(run_condition(model, tokenizer, prompts,
                                beta_ratio=1.0, beta_layers=None, target_layers=target_layers,
                                gen_tokens=args.gen_tokens, metric_max_tokens=args.metric_max_tokens,
                                label="baseline"))
    frames.append(run_condition(model, tokenizer, prompts,
                                beta_ratio=args.beta, beta_layers=target_layers, target_layers=target_layers,
                                gen_tokens=args.gen_tokens, metric_max_tokens=args.metric_max_tokens,
                                label=f"beta_{args.beta:g}"))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(outdir / "per_prompt.csv", index=False)

    cols = ["degree_entropy", "clustering", "global_efficiency",
            "associative_drift", "perplexity", "local_coherence"]
    agg = df.groupby("condition")[cols].mean().reset_index()
    agg.to_csv(outdir / "summary_by_condition.csv", index=False)

    base = agg[agg["condition"] == "baseline"].iloc[0]
    beta = agg[agg["condition"].str.startswith("beta_")].iloc[0]
    def delta(c): return float(beta[c] - base[c])

    shifts = {c: delta(c) for c in ["degree_entropy", "clustering", "global_efficiency"]}
    # EBH-paper signature: degree entropy UP, clustering (local integ) UP,
    # global efficiency (global integ) DOWN.
    matches_paper = (shifts["degree_entropy"] > 0 and
                     shifts["clustering"] > 0 and
                     shifts["global_efficiency"] < 0)
    moved = abs(shifts["degree_entropy"]) > 0.02

    verdict = {
        "behavioral_effect": {"drift_shift": delta("associative_drift"),
                              "perplexity_shift": delta("perplexity")},
        "graph_metric_shifts_baseline_to_beta": shifts,
        "degree_entropy_moved": bool(moved),
        "matches_ayahuasca_signature": bool(matches_paper),
        "interpretation": (
            ("Degree-distribution entropy moved under beta"
             + (" AND the full signature (entropy up, local integration up, global "
                "integration down) matches the ayahuasca paper. Strong screening "
                "result -- worth building a direct intervention on degree entropy."
                if matches_paper else
                ", but the local/global integration pattern does NOT match the "
                "ayahuasca signature. Partial; inspect which companion metric diverges."))
            if moved else
            "Degree-distribution entropy did not move meaningfully under beta. Like "
            "the other entropy measures, this may not be where beta's effect lives. "
            "Inspect per-density values before concluding."),
        "caveats": [
            "Screening only; does NOT establish causation. The advantage of THIS "
            "measure is it is interveneable -- a direct degree-entropy intervention "
            "is the causal test.",
            "Attention graph is directed/per-pass/task-driven; brain FC is "
            "undirected/resting. Symmetrized here to approximate, but semantics differ.",
            "Compared at matched edge density across a density sweep to avoid the "
            "threshold artifact the paper flagged.",
            "n=%d prompts; directional only." % args.num_prompts,
        ],
    }

    summary = {"model": args.model, "beta": args.beta, "target_layers": target_layers,
               "densities": list(DENSITIES), "num_prompts": args.num_prompts,
               "verdict": verdict,
               "files": {"per_prompt": "per_prompt.csv",
                         "summary_by_condition": "summary_by_condition.csv"}}
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\n=== ATTENTION-GRAPH DEGREE-ENTROPY SCREEN ===")
    print(json.dumps(summary, indent=2, default=float))
    print(f"\nWrote results to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
