#!/usr/bin/env python3
"""
Beta/layer sweep for Llama attention inverse-temperature interventions,
reframed around the entropic-brain / Hopfield analogy.

Default model:
    meta-llama/Llama-3.2-1B-Instruct

Thesis
------
Under the modern-Hopfield reading of attention (Ramsauer et al.), the softmax
inverse-temperature beta controls the sharpness of the retrieval energy
landscape: high beta -> sharp isolated basins (one stored pattern retrieved),
low beta -> broad merged basins (metastable blends of patterns). The REBUS /
entropic-brain models describe psychedelics as flattening high-level priors and
raising the entropy of dynamics. This harness tests whether flattening the
attention landscape produces an *analogous information-processing signature* in
an LLM: associative drift rising while local coherence is still preserved, then
coherence collapsing past some "dose."

This is an analogy between two complex systems that share an abstract
description, NOT a claim that the model is "on" psychedelics.

What it does
------------
1. Generates OPEN-ENDED prompts (continuations, associations, descriptions,
   loose Q&A) where associative blending is the *output*, not noise.
2. Patches Llama eager attention so selected layers use
       softmax(beta_ratio * QK^T / sqrt(d))
   beta_ratio=1.0 is baseline; lower flattens the landscape.
3. Runs:
   - layer sweep: flatten one layer at a time
   - beta sweep: flatten selected layers across a dense beta grid
   - CONTROL sweep: vary sampling temperature instead of beta, to show the
     two knobs produce distinguishable signatures (the key skeptic rebuttal).
4. Measures, per generation:
   - local_coherence: clean-copy perplexity of the model's own output
     (grammatical/local well-formedness; lower ppl -> more coherent)
   - associative_drift: how far the continuation wanders from the prompt,
     via embedding cosine distance (semantic loosening)
   - self_diversity: internal lexical/semantic diversity of the output
   - global_coherence: topic connectivity across the output (does it go
     anywhere, or just float)
   - attention_entropy: normalized entropy of attention maps
5. Detects the "interesting band": beta range where drift/diversity are
   elevated but local coherence is still intact -- the altered-but-coherent
   regime, which is the actual analog you care about.
6. Writes CSVs, PNG plots (incl. the dual drift-vs-coherence curve and the
   beta-vs-temperature comparison), and summary.json.

Setup
-----
    pip install torch transformers accelerate pandas matplotlib tqdm numpy

Example
-------
    python beta_psychedelic_sweep_llama.py \
      --num-prompts 24 \
      --gen-tokens 96 \
      --samples-per-prompt 3 \
      --beta-values 1.0,0.9,0.8,0.7,0.6,0.5,0.42,0.35,0.28,0.22,0.16,0.1 \
      --temp-values 0.7,1.0,1.3,1.6,2.0,2.5 \
      --outdir psyche_sweep_results

Notes
-----
- The beta arm uses do_sample=False so the *only* source of variation is the
  attention-landscape intervention (a causal study).
- The temperature control arm uses beta_ratio=1.0 and varies do_sample
  temperature, isolating output-distribution entropy from landscape entropy.
- Define metrics BEFORE looking at outputs; don't pattern-match dreaminess onto
  whatever you get.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------
# Global intervention state
# -----------------------------

@dataclass
class InterventionState:
    active: bool = False
    beta_ratio: float = 1.0
    layers: Optional[set[int]] = None

    def ratio_for_layer(self, layer_idx: int) -> float:
        if not self.active:
            return 1.0
        if self.layers is None or layer_idx in self.layers:
            return self.beta_ratio
        return 1.0

INTERVENTION = InterventionState()

# Diagnostic: count how many times the patched attention actually fires with a
# non-trivial ratio. If this stays 0 during an intervention, the patch is not
# wired into the dispatch path your model is using.
PATCH_CALLS = {"total": 0, "intervened": 0}


def patch_llama_attention() -> None:
    """Patch HF Llama eager attention to multiply the attention scale per layer."""
    from transformers.models.llama import modeling_llama

    if getattr(modeling_llama.eager_attention_forward, "_beta_sweep_patched", False):
        return

    original = modeling_llama.eager_attention_forward

    def patched(module, query, key, value, attention_mask, scaling, **kwargs):
        layer_idx = getattr(module, "layer_idx", 0)
        r = INTERVENTION.ratio_for_layer(layer_idx)
        PATCH_CALLS["total"] += 1
        if r != 1.0:
            PATCH_CALLS["intervened"] += 1
        return original(module, query, key, value, attention_mask, scaling * r, **kwargs)

    patched._beta_sweep_patched = True  # type: ignore[attr-defined]
    patched._beta_sweep_original = original  # type: ignore[attr-defined]
    modeling_llama.eager_attention_forward = patched

    # Modern transformers dispatch through a registry. Patch known locations.
    patched_registry = False
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS["eager"] = patched
        patched_registry = True
    except Exception:
        pass

    if not patched_registry:
        try:
            from transformers.modeling_utils import AttentionInterface
            if hasattr(AttentionInterface, "_global_mapping"):
                AttentionInterface._global_mapping["eager"] = patched
                patched_registry = True
            elif hasattr(AttentionInterface, "register"):
                AttentionInterface.register("eager", patched)
                patched_registry = True
        except Exception:
            pass

    if not patched_registry:
        print("WARNING: could not patch the attention registry. The module symbol was patched, "
              "but your transformers version may dispatch elsewhere.")


def assert_patch_live(model, tokenizer) -> None:
    """Run a tiny forward under a real intervention and confirm the patch fired.

    This is the single most important sanity check: it catches the silent
    'intervention never actually ran' failure mode (wrong attn impl, dispatch
    elsewhere, context manager not entered).
    """
    impl = getattr(model.config, "_attn_implementation", None) or getattr(
        model.config, "attn_implementation", None
    )
    if impl not in (None, "eager"):
        print(f"WARNING: model attn_implementation is '{impl}', not 'eager'. "
              "The eager patch will not be on the active code path. "
              "Reload with attn_implementation='eager'.")
    before = PATCH_CALLS["intervened"]
    enc = tokenizer("probe sentence for patch check", return_tensors="pt").to(model.device)
    with torch.no_grad(), beta_intervention(beta_ratio=0.5, layers=[0]):
        model(**enc, use_cache=False)
    fired = PATCH_CALLS["intervened"] - before
    if fired == 0:
        raise RuntimeError(
            "Beta patch did NOT fire during a forward pass. The intervention is "
            "inactive: your transformers version is dispatching attention somewhere "
            "the patch did not reach, or the model is not using eager attention. "
            "Fix this before trusting any sweep result."
        )
    print(f"[patch check] OK - patched attention fired {fired} times under intervention.")


@contextmanager
def beta_intervention(beta_ratio: float = 1.0, layers: Optional[Iterable[int]] = None):
    old = InterventionState(INTERVENTION.active, INTERVENTION.beta_ratio, INTERVENTION.layers)
    INTERVENTION.active = beta_ratio != 1.0
    INTERVENTION.beta_ratio = float(beta_ratio)
    INTERVENTION.layers = None if layers is None else set(int(x) for x in layers)
    try:
        yield
    finally:
        INTERVENTION.active = old.active
        INTERVENTION.beta_ratio = old.beta_ratio
        INTERVENTION.layers = old.layers


# -----------------------------
# Open-ended prompt generation
# -----------------------------
# These are designed so that associative blending shows up *in the output*.
# There is no single correct answer; the dependent variables are coherence and
# drift, which can move smoothly.

CONTINUATION_SEEDS = [
    "The last train of the evening pulled out of the station, and",
    "She opened the box she had been avoiding for years, and inside",
    "On the morning the river changed direction, the town",
    "He set the kettle on, looked out at the rain, and thought about",
    "The map showed a road that no one in the village remembered building, so",
    "When the lights came back on after the storm, the room",
]
ASSOCIATION_SEEDS = [
    "lighthouse", "clockwork", "saltwater", "threshold", "ember",
    "library", "migration", "mirror", "harvest", "static",
]
DESCRIPTION_SEEDS = [
    "an abandoned greenhouse at dusk",
    "the inside of a seashell",
    "a city seen from a plane at night",
    "a kitchen the morning after a party",
    "a forest path in early autumn",
    "a desk belonging to someone who has just left",
]
LOOSE_QA = [
    "What does a quiet afternoon feel like?",
    "If memory had a texture, what would it be?",
    "Why do old buildings seem to hold the weather?",
    "What is the relationship between a road and the place it leaves?",
    "How would you describe the color of waiting?",
    "What happens to a song after it ends?",
]


def make_open_prompts(n: int, seed: int) -> List[Dict[str, str]]:
    """Build a balanced mix of open-ended prompt types."""
    rng = random.Random(seed)
    pool: List[Tuple[str, str]] = []
    for s in CONTINUATION_SEEDS:
        pool.append(("continuation", f"Continue this passage in a few sentences:\n\n{s}"))
    for w in ASSOCIATION_SEEDS:
        pool.append(("association",
                     f"Free-associate from the word '{w}'. Write a short chain of "
                     f"connected images and ideas, a sentence or two."))
    for d in DESCRIPTION_SEEDS:
        pool.append(("description", f"Describe {d} in a few vivid sentences."))
    for q in LOOSE_QA:
        pool.append(("loose_qa", q))

    rng.shuffle(pool)
    if n <= len(pool):
        chosen = pool[:n]
    else:
        chosen = [pool[i % len(pool)] for i in range(n)]

    prompts = []
    for i, (kind, text) in enumerate(chosen):
        prompts.append({"prompt_id": f"p{i:03d}", "kind": kind, "prompt": text})
    return prompts


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


# -----------------------------
# Generation
# -----------------------------

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    seed: Optional[int] = None,
) -> str:
    if seed is not None:
        torch.manual_seed(seed)
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    eos_ids = [tokenizer.eos_token_id]
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot is not None and eot != tokenizer.unk_token_id:
        eos_ids.append(eot)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_ids,
        pad_token_id=tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
    else:
        gen_kwargs.update(do_sample=False)

    out = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(out[0, input_len:], skip_special_tokens=True).strip()


# -----------------------------
# Metrics
# -----------------------------
# IMPORTANT: all coherence metrics are computed under a CLEAN model state
# (no intervention). We measure the *artifact* the altered model produced, not
# the altered model's opinion of it. Otherwise a flattened model would rate its
# own degraded output as fine.

@torch.no_grad()
def clean_perplexity(model, tokenizer, text: str, max_tokens: int) -> float:
    """Perplexity of `text` under the UN-intervened model. Lower = more locally
    coherent / grammatical. This is our local_coherence signal (inverted)."""
    text = text.strip()
    if len(text.split()) < 2:
        return float("nan")
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens).to(model.device)
    ids = enc["input_ids"]
    if ids.shape[1] < 2:
        return float("nan")
    # Ensure no intervention is active while scoring.
    with beta_intervention(beta_ratio=1.0, layers=None):
        out = model(**enc, use_cache=False)
    logits = out.logits[:, :-1, :].float()
    targets = ids[:, 1:]
    logp = F.log_softmax(logits, dim=-1)
    tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    nll = -tok_logp.mean().item()
    return float(math.exp(min(nll, 20.0)))  # clamp to avoid inf on garbage


@torch.no_grad()
def mean_pool_embedding(model, tokenizer, text: str, max_tokens: int) -> Optional[torch.Tensor]:
    text = text.strip()
    if not text:
        return None
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens).to(model.device)
    with beta_intervention(beta_ratio=1.0, layers=None):
        out = model(**enc, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[-1].float()[0]
    mask = enc["attention_mask"][0].bool()
    emb = h[mask].mean(dim=0)
    return F.normalize(emb, dim=0).cpu()


@torch.no_grad()
def associative_drift(model, tokenizer, prompt: str, output: str, max_tokens: int) -> float:
    """Cosine distance between prompt and output embeddings. Higher = output
    wanders further from the prompt (semantic loosening)."""
    if not output.strip():
        return 1.0
    e1 = mean_pool_embedding(model, tokenizer, prompt, max_tokens)
    e2 = mean_pool_embedding(model, tokenizer, output, max_tokens)
    if e1 is None or e2 is None:
        return 1.0
    return float(1.0 - torch.dot(e1, e2).item())


def lexical_diversity(text: str) -> float:
    """Type-token ratio over a fixed window. Higher = more varied vocabulary."""
    toks = re.findall(r"\w+", text.lower())
    if not toks:
        return 0.0
    window = toks[:120]
    return len(set(window)) / len(window)


@torch.no_grad()
def global_coherence(model, tokenizer, output: str, max_tokens: int) -> float:
    """Mean pairwise cosine SIMILARITY between consecutive sentence embeddings.
    High = sentences stay on topic (the piece goes somewhere). Low = each
    sentence floats off on its own (dreamy/disconnected global structure).

    The psychedelic-analog signature is: local_coherence preserved (low ppl)
    while global_coherence drops -- sentences are individually fine but the
    paragraph doesn't cohere.
    """
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", output.strip()) if len(s.split()) >= 3]
    if len(sents) < 2:
        return float("nan")
    embs = []
    for s in sents[:8]:
        e = mean_pool_embedding(model, tokenizer, s, max_tokens)
        if e is not None:
            embs.append(e)
    if len(embs) < 2:
        return float("nan")
    sims = []
    for a, b in zip(embs[:-1], embs[1:]):
        sims.append(float(torch.dot(a, b).item()))
    return float(sum(sims) / len(sims))


def output_health(text: str) -> Dict[str, float]:
    """Cheap degenerate-output guards (runaway repetition / empty)."""
    toks = re.findall(r"\w+", text.lower())
    if not toks:
        return {"out_tokens": 0, "repeat_frac": 1.0, "degenerate": 1.0}
    most_common = max(toks.count(t) for t in set(toks))
    repeat_frac = most_common / len(toks)
    degenerate = 1.0 if (repeat_frac > 0.5 and len(toks) > 8) else 0.0
    return {"out_tokens": float(len(toks)), "repeat_frac": repeat_frac, "degenerate": degenerate}


@torch.no_grad()
def attention_entropy_for_text(model, tokenizer, text: str, max_tokens: int) -> Tuple[float, Dict[int, float]]:
    """Normalized attention entropy under the CLEAN model (descriptive readout
    of the artifact, not of the intervened pass)."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens).to(model.device)
    with beta_intervention(beta_ratio=1.0, layers=None):
        out = model(**enc, output_attentions=True, use_cache=False)
    attentions = out.attentions
    if attentions is None:
        return float("nan"), {}
    per_layer = {}
    vals = []
    eps = 1e-12
    for layer_idx, attn in enumerate(attentions):
        p = attn.float().clamp_min(eps)
        ent = -(p * p.log()).sum(dim=-1)
        key_len = p.shape[-1]
        norm = math.log(max(key_len, 2))
        ent_norm = (ent / norm).mean().item()
        per_layer[layer_idx] = ent_norm
        vals.append(ent_norm)
    return float(sum(vals) / len(vals)), per_layer


# -----------------------------
# Experiment runner
# -----------------------------

@torch.no_grad()
def evaluate_condition(
    model,
    tokenizer,
    prompts: Sequence[Dict[str, str]],
    condition: str,
    *,
    beta_ratio: float,
    layers: Optional[Iterable[int]],
    do_sample: bool,
    temperature: float,
    samples_per_prompt: int,
    gen_tokens: int,
    metric_max_tokens: int,
    base_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    entropy_rows = []
    layer_list = None if layers is None else list(layers)
    layers_label = "all" if layer_list is None else ",".join(map(str, layer_list))

    for item in tqdm(prompts, desc=condition, leave=False):
        for s in range(samples_per_prompt):
            # Generation happens UNDER the intervention.
            with beta_intervention(beta_ratio=beta_ratio, layers=layer_list):
                gen_seed = base_seed + 1000 * s + int(item["prompt_id"][1:])
                output = generate(
                    model, tokenizer, item["prompt"], gen_tokens,
                    do_sample=do_sample, temperature=temperature,
                    seed=gen_seed if do_sample else None,
                )

            # Metrics computed under CLEAN model (handled inside each metric).
            ppl = clean_perplexity(model, tokenizer, output, metric_max_tokens)
            drift = associative_drift(model, tokenizer, item["prompt"], output, metric_max_tokens)
            div = lexical_diversity(output)
            gcoh = global_coherence(model, tokenizer, output, metric_max_tokens)
            ent_mean, ent_by_layer = attention_entropy_for_text(
                model, tokenizer, item["prompt"] + "\n" + output, metric_max_tokens
            )
            health = output_health(output)
            # local_coherence in [0,1]-ish: 1 at ppl<=10, decaying with log-ppl.
            local_coh = float(1.0 / (1.0 + max(0.0, math.log(max(ppl, 1e-6)) - math.log(10.0)))) \
                if not math.isnan(ppl) else float("nan")

            rows.append({
                "condition": condition,
                "prompt_id": item["prompt_id"],
                "kind": item["kind"],
                "sample": s,
                "arm": "temperature" if (do_sample and beta_ratio == 1.0) else "beta",
                "beta_ratio": beta_ratio,
                "temperature": temperature if do_sample else 0.0,
                "layers": layers_label,
                "output": output,
                "perplexity": ppl,
                "local_coherence": local_coh,
                "associative_drift": drift,
                "self_diversity": div,
                "global_coherence": gcoh,
                "attention_entropy": ent_mean,
                **health,
            })
            for li, ev in ent_by_layer.items():
                entropy_rows.append({
                    "condition": condition,
                    "prompt_id": item["prompt_id"],
                    "sample": s,
                    "beta_ratio": beta_ratio,
                    "temperature": temperature if do_sample else 0.0,
                    "intervened_layers": layers_label,
                    "attention_layer": li,
                    "attention_entropy": ev,
                })
    return pd.DataFrame(rows), pd.DataFrame(entropy_rows)


def aggregate(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    return df.groupby(group_cols, dropna=False).agg(
        local_coherence=("local_coherence", "mean"),
        local_coherence_sd=("local_coherence", "std"),
        associative_drift=("associative_drift", "mean"),
        associative_drift_sd=("associative_drift", "std"),
        self_diversity=("self_diversity", "mean"),
        global_coherence=("global_coherence", "mean"),
        attention_entropy=("attention_entropy", "mean"),
        perplexity=("perplexity", "mean"),
        degenerate=("degenerate", "mean"),
        n=("local_coherence", "count"),
    ).reset_index()


# -----------------------------
# Analysis
# -----------------------------

def detect_strange_layers(layer_summary: pd.DataFrame, baseline: Dict[str, float]) -> pd.DataFrame:
    """Rank layers by how much flattening them produces the target signature:
    drift/diversity up, attention entropy up, while local coherence is held and
    output is not degenerate."""
    df = layer_summary.copy()
    base_drift = baseline.get("associative_drift", 0.0)
    base_div = baseline.get("self_diversity", 0.0)
    base_ent = baseline.get("attention_entropy", 0.0)
    base_coh = max(baseline.get("local_coherence", 1e-6), 1e-6)
    df["drift_increase"] = df["associative_drift"] - base_drift
    df["diversity_increase"] = df["self_diversity"] - base_div
    df["entropy_increase"] = df["attention_entropy"] - base_ent
    df["coherence_retained"] = (df["local_coherence"] / base_coh).clip(upper=1.0)

    df["strange_score"] = (
        1.5 * df["drift_increase"].clip(lower=0)
        + 1.0 * df["diversity_increase"].clip(lower=0)
        + 1.0 * df["entropy_increase"].clip(lower=0)
    ) * df["coherence_retained"] * (1.0 - df["degenerate"].fillna(0))

    thresh = df["strange_score"].quantile(0.70)
    df["strange_candidate"] = (
        (df["strange_score"] > thresh)
        & (df["coherence_retained"] >= 0.6)
        & (df["degenerate"].fillna(0) < 0.25)
    )
    return df.sort_values("strange_score", ascending=False)


def detect_interesting_band(beta_summary: pd.DataFrame, baseline: Dict[str, float]) -> Dict[str, object]:
    """Find the altered-but-coherent band: beta values where associative_drift
    is meaningfully elevated over baseline AND local_coherence is still mostly
    intact AND output is not degenerate. This replaces the old accuracy-cliff
    'beta_star' framing, which measured the wrong thing for this task."""
    df = beta_summary.sort_values("beta_ratio", ascending=False).reset_index(drop=True)
    if len(df) < 3:
        return {"band": None, "reason": "Need at least 3 beta values."}

    base_drift = baseline.get("associative_drift", 0.0)
    base_coh = baseline.get("local_coherence", 1.0)
    drift_sd = max(baseline.get("associative_drift_sd", 0.0) or 0.0, 1e-6)

    # "Elevated" = drift at least ~1 baseline-SD above baseline.
    df["drift_elevated"] = (df["associative_drift"] - base_drift) >= drift_sd
    # "Coherent" = local coherence retained >= 70% of baseline, not degenerate.
    df["coherent"] = (df["local_coherence"] >= 0.7 * base_coh) & (df["degenerate"].fillna(0) < 0.25)
    df["in_band"] = df["drift_elevated"] & df["coherent"]

    band = df[df["in_band"]]["beta_ratio"].astype(float).to_list()
    # The "collapse" beta = highest beta (least flattening) where coherence
    # first falls below the threshold as beta decreases.
    collapse_beta = None
    for _, r in df.iterrows():
        if not r["coherent"]:
            collapse_beta = float(r["beta_ratio"])
            break

    sweet_spot = max(band) if band else None  # gentlest dose that still shows the effect
    return {
        "band_beta_values": band,
        "sweet_spot_beta": sweet_spot,
        "collapse_beta": collapse_beta,
        "baseline_drift": base_drift,
        "baseline_local_coherence": base_coh,
        "found_band": bool(band),
        "interpretation": (
            "altered-but-coherent band found: drift elevated while local coherence held"
            if band else
            "no band by default thresholds; widen beta grid lower or relax thresholds"
        ),
    }


def compare_beta_vs_temperature(beta_summary: pd.DataFrame, temp_summary: pd.DataFrame) -> Dict[str, object]:
    """Are the two knobs distinguishable? The skeptic objection is 'you just
    reinvented high sampling temperature.' We compare the drift/coherence and
    global-coherence profiles. If beta drops GLOBAL coherence and raises
    attention entropy more than temperature does at matched drift, the effect is
    mechanistically distinct."""
    if beta_summary.empty or temp_summary.empty:
        return {"comparable": False, "reason": "missing one of the two arms"}

    def profile(df):
        return {
            "max_drift": float(df["associative_drift"].max()),
            "min_local_coherence": float(df["local_coherence"].min()),
            "min_global_coherence": float(df["global_coherence"].min(skipna=True)),
            "max_attention_entropy": float(df["attention_entropy"].max()),
        }

    b = profile(beta_summary)
    t = profile(temp_summary)
    # Attention entropy is the cleanest discriminator: beta acts ON attention,
    # temperature acts on the output sampling distribution and should move
    # attention entropy far less.
    entropy_gap = b["max_attention_entropy"] - t["max_attention_entropy"]
    global_gap = t["min_global_coherence"] - b["min_global_coherence"]
    return {
        "comparable": True,
        "beta_profile": b,
        "temperature_profile": t,
        "attention_entropy_gap_beta_minus_temp": entropy_gap,
        "global_coherence_drop_gap_beta_minus_temp": global_gap,
        "distinguishable": bool(entropy_gap > 0.02 or global_gap > 0.05),
        "interpretation": (
            "beta flattening moves attention entropy / global coherence more than "
            "temperature at comparable drift -> mechanistically distinct knob"
            if (entropy_gap > 0.02 or global_gap > 0.05) else
            "beta and temperature look similar on these metrics here; the effect "
            "may not be specific to the attention landscape -- investigate"
        ),
    }


# -----------------------------
# Plots
# -----------------------------

def make_plots(layer_summary, beta_summary, temp_summary, outdir: Path) -> None:
    import matplotlib.pyplot as plt

    if not layer_summary.empty:
        x = layer_summary["layer"].astype(int)
        for metric in ["local_coherence", "associative_drift", "self_diversity",
                       "global_coherence", "attention_entropy", "strange_score"]:
            if metric not in layer_summary:
                continue
            plt.figure()
            plt.plot(x, layer_summary[metric], marker="o")
            plt.xlabel("Intervened layer")
            plt.ylabel(metric)
            plt.title(f"Layer sweep: {metric}")
            plt.tight_layout()
            plt.savefig(outdir / f"layer_sweep_{metric}.png", dpi=160)
            plt.close()

    if not beta_summary.empty:
        x = beta_summary["beta_ratio"].astype(float)
        for metric in ["local_coherence", "associative_drift", "self_diversity",
                       "global_coherence", "attention_entropy"]:
            plt.figure()
            plt.plot(x, beta_summary[metric], marker="o")
            plt.gca().invert_xaxis()
            plt.xlabel("beta_ratio (lower = flatter)")
            plt.ylabel(metric)
            plt.title(f"Beta sweep: {metric}")
            plt.tight_layout()
            plt.savefig(outdir / f"beta_sweep_{metric}.png", dpi=160)
            plt.close()

        # The money plot: drift vs local coherence on one axis set.
        fig, ax1 = plt.subplots()
        ax1.plot(x, beta_summary["associative_drift"], marker="o", color="tab:red", label="associative drift")
        ax1.set_xlabel("beta_ratio (lower = flatter)")
        ax1.set_ylabel("associative drift", color="tab:red")
        ax1.invert_xaxis()
        ax2 = ax1.twinx()
        ax2.plot(x, beta_summary["local_coherence"], marker="s", color="tab:blue", label="local coherence")
        ax2.set_ylabel("local coherence", color="tab:blue")
        plt.title("Beta sweep: drift rises while coherence holds, then falls")
        fig.tight_layout()
        plt.savefig(outdir / "beta_sweep_drift_vs_coherence.png", dpi=160)
        plt.close()

    # Beta vs temperature control comparison (drift on x, attention entropy on y).
    if not beta_summary.empty and not temp_summary.empty:
        plt.figure()
        plt.scatter(beta_summary["associative_drift"], beta_summary["attention_entropy"],
                    marker="o", label="beta arm")
        plt.scatter(temp_summary["associative_drift"], temp_summary["attention_entropy"],
                    marker="^", label="temperature arm")
        plt.xlabel("associative drift")
        plt.ylabel("attention entropy")
        plt.title("Control: beta vs temperature in (drift, attention-entropy) space")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "control_beta_vs_temperature.png", dpi=160)
        plt.close()


def parse_floats(s: str, ensure: Optional[float] = None, reverse: bool = True) -> List[float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if ensure is not None and ensure not in vals:
        vals = [ensure] + vals
    return sorted(set(vals), reverse=reverse)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-1B-Instruct"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--outdir", default="psyche_sweep_results")
    parser.add_argument("--num-prompts", type=int, default=18)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gen-tokens", type=int, default=96)
    parser.add_argument("--samples-per-prompt", type=int, default=3,
                        help="repeats per prompt; >1 useful in the temperature arm")
    parser.add_argument("--metric-max-tokens", type=int, default=1024)
    parser.add_argument("--layer-beta", type=float, default=0.35,
                        help="beta_ratio used for one-layer-at-a-time sweep")
    parser.add_argument("--beta-values", default="1.0,0.9,0.8,0.7,0.6,0.5,0.42,0.35,0.28,0.22,0.16,0.1")
    parser.add_argument("--temp-values", default="0.7,1.0,1.3,1.6,2.0,2.5",
                        help="sampling temperatures for the control arm (beta=1.0)")
    parser.add_argument("--beta-layers", default="auto",
                        help="comma layers for beta sweep, 'auto' for strange candidates, or 'all'")
    parser.add_argument("--skip-layer-sweep", action="store_true")
    parser.add_argument("--skip-beta-sweep", action="store_true")
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
        args.model,
        torch_dtype=dtype,
        device_map=device if device != "cpu" else None,
        attn_implementation="eager",
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()
    patch_llama_attention()
    assert_patch_live(model, tokenizer)  # fail loudly if the intervention is inert

    prompts = make_open_prompts(args.num_prompts, args.seed)
    with open(outdir / "prompts.jsonl", "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    all_rows = []
    all_entropy_rows = []

    # ---- Baseline (no intervention, greedy) ----
    base_df, base_ent = evaluate_condition(
        model, tokenizer, prompts, "baseline",
        beta_ratio=1.0, layers=None, do_sample=False, temperature=0.0,
        samples_per_prompt=1, gen_tokens=args.gen_tokens,
        metric_max_tokens=args.metric_max_tokens, base_seed=args.seed,
    )
    all_rows.append(base_df)
    all_entropy_rows.append(base_ent)
    baseline_summary = aggregate(base_df, ["condition"]).iloc[0].to_dict()

    n_layers = int(model.config.num_hidden_layers)
    layer_summary = pd.DataFrame()
    strange_layers: List[int] = []

    # ---- Layer sweep ----
    if not args.skip_layer_sweep:
        for layer in range(n_layers):
            df, ent = evaluate_condition(
                model, tokenizer, prompts, f"layer_{layer}",
                beta_ratio=args.layer_beta, layers=[layer], do_sample=False, temperature=0.0,
                samples_per_prompt=1, gen_tokens=args.gen_tokens,
                metric_max_tokens=args.metric_max_tokens, base_seed=args.seed,
            )
            all_rows.append(df)
            all_entropy_rows.append(ent)

        full_df = pd.concat(all_rows, ignore_index=True)
        layer_df = full_df[full_df["condition"].str.startswith("layer_")].copy()
        layer_df["layer"] = layer_df["condition"].str.extract(r"layer_(\d+)").astype(int)
        layer_summary = aggregate(layer_df, ["layer"])
        layer_summary = detect_strange_layers(layer_summary, baseline_summary)
        layer_summary.to_csv(outdir / "layer_summary.csv", index=False)
        strange_layers = layer_summary[layer_summary["strange_candidate"]]["layer"].astype(int).to_list()

    # ---- Choose beta-sweep layers ----
    beta_layers: Optional[List[int]]
    if args.beta_layers == "all":
        beta_layers = None
    elif args.beta_layers == "auto":
        if strange_layers:
            beta_layers = strange_layers
        elif not layer_summary.empty:
            beta_layers = layer_summary.head(max(1, min(4, len(layer_summary))))["layer"].astype(int).to_list()
        else:
            beta_layers = None
    else:
        beta_layers = [int(x.strip()) for x in args.beta_layers.split(",") if x.strip()]

    # ---- Beta sweep (causal: greedy, vary beta) ----
    beta_summary = pd.DataFrame()
    band = {"found_band": False, "reason": "beta sweep skipped"}
    if not args.skip_beta_sweep:
        beta_values = parse_floats(args.beta_values, ensure=1.0)
        for b in beta_values:
            df, ent = evaluate_condition(
                model, tokenizer, prompts, f"beta_{b:g}",
                beta_ratio=b, layers=beta_layers, do_sample=False, temperature=0.0,
                samples_per_prompt=1, gen_tokens=args.gen_tokens,
                metric_max_tokens=args.metric_max_tokens, base_seed=args.seed,
            )
            all_rows.append(df)
            all_entropy_rows.append(ent)
        full_df = pd.concat(all_rows, ignore_index=True)
        beta_df = full_df[full_df["condition"].str.startswith("beta_")].copy()
        beta_summary = aggregate(beta_df, ["beta_ratio", "layers"])
        beta_summary.to_csv(outdir / "beta_summary.csv", index=False)
        band = detect_interesting_band(beta_summary, baseline_summary)

    # ---- Temperature control (beta=1.0, vary sampling temperature) ----
    temp_summary = pd.DataFrame()
    if not args.skip_temp_control:
        temp_values = parse_floats(args.temp_values, reverse=False)
        for t in temp_values:
            df, ent = evaluate_condition(
                model, tokenizer, prompts, f"temp_{t:g}",
                beta_ratio=1.0, layers=None, do_sample=True, temperature=t,
                samples_per_prompt=args.samples_per_prompt, gen_tokens=args.gen_tokens,
                metric_max_tokens=args.metric_max_tokens, base_seed=args.seed,
            )
            all_rows.append(df)
            all_entropy_rows.append(ent)
        full_df = pd.concat(all_rows, ignore_index=True)
        temp_df = full_df[full_df["condition"].str.startswith("temp_")].copy()
        temp_summary = aggregate(temp_df, ["temperature"])
        temp_summary.to_csv(outdir / "temp_summary.csv", index=False)

    control = compare_beta_vs_temperature(beta_summary, temp_summary)

    # ---- Persist ----
    results_df = pd.concat(all_rows, ignore_index=True)
    entropy_df = pd.concat(all_entropy_rows, ignore_index=True)
    results_df.to_csv(outdir / "per_generation_results.csv", index=False)
    entropy_df.to_csv(outdir / "per_layer_attention_entropy.csv", index=False)

    make_plots(layer_summary, beta_summary, temp_summary, outdir)

    summary = {
        "model": args.model,
        "n_layers": n_layers,
        "num_prompts": args.num_prompts,
        "gen_tokens": args.gen_tokens,
        "samples_per_prompt": args.samples_per_prompt,
        "layer_beta_ratio": args.layer_beta,
        "patch_calls": PATCH_CALLS,
        "baseline": baseline_summary,
        "strange_layers": strange_layers,
        "beta_sweep_layers": "all" if beta_layers is None else beta_layers,
        "interesting_band": band,
        "beta_vs_temperature_control": control,
        "metric_notes": {
            "local_coherence": "clean-model perplexity-derived; higher = more grammatical/local well-formedness",
            "associative_drift": "1 - cos(prompt, output) under clean model; higher = wanders further",
            "self_diversity": "type-token ratio over first 120 tokens",
            "global_coherence": "mean consecutive-sentence cosine similarity; LOW = floaty/disconnected",
            "signature": "psychedelic-analog = drift/diversity up + local_coherence held + global_coherence down",
        },
        "files": {
            "prompts": "prompts.jsonl",
            "per_generation_results": "per_generation_results.csv",
            "per_layer_attention_entropy": "per_layer_attention_entropy.csv",
            "layer_summary": "layer_summary.csv" if not layer_summary.empty else None,
            "beta_summary": "beta_summary.csv" if not beta_summary.empty else None,
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
