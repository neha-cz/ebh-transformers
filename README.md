# Attention Inverse-Temperature in Transformers: Testing the Entropic Brain Hypothesis

Mechanistic study of how attention inverse-temperature (β) shapes generation in
Llama-3.2-1B-Instruct, framed against neuroscientific models of the psychedelic
state (the Entropic Brain Hypothesis, EBH, and REBUS).

## Motivation

Under the Hopfield interpretation of attention, lowering β flattens the attention
energy landscape (blurs attention sharpness). This parallels REBUS's "flattening
of the variational free-energy landscape" under psychedelics (Carhart-Harris &
Friston, 2019). Since β is literally an inverse-temperature, lowering it
implements that temperature-raising move in attention space.

**Hypothesis:** flattening the landscape via β causally produces loosened,
psychedelic-like reasoning by raising entropy (an EBH-style prediction).

## Methods

**The β intervention.** β scales the query–key attention logits before softmax.
We patch Llama's eager attention so that, on selected layers, the logit scaling
is multiplied by a ratio < 1 (default 0.45), flattening the attention
distribution. The patch is a context manager that can be toggled per layer and
per forward pass, leaving all model weights unchanged — the intervention is
purely at inference time. (Requires `attn_implementation="eager"`.)

**Prompt set (n = 8 open-ended prompts, fixed seed).** Reasoning was probed with
open-ended generation rather than fixed-answer tasks, since the hypothesis is
about *loosening* and associative drift, which closed tasks can't reveal. The
prompt pool is a balanced mix of four types:

- **Continuation** — finish an evocative passage (e.g. "The last train of the
  evening pulled out of the station, and…")
- **Free-association** — chain images from a seed word (lighthouse, clockwork,
  threshold, ember…)
- **Description** — describe a scene vividly (an abandoned greenhouse at dusk,
  the inside of a seashell…)
- **Loose Q&A** — open reflective questions ("If memory had a texture, what would
  it be?", "What happens to a song after it ends?")

**Defining "reasoning" — two metrics.** We operationalized the altered state
along two axes, both scored under the *un-intervened* model so the measurement
itself is never distorted by β:

- **Associative drift** = `1 − cosine_similarity(embed(prompt), embed(output))`,
  using mean-pooled last-layer hidden states. Higher drift = output wanders
  semantically further from the prompt (associative loosening). This captures the
  "loosened/dreamy" axis.
- **Local coherence**, derived from **clean perplexity**: perplexity of the
  generated text under the unmodified model (`exp` of mean token NLL, clamped).
  Lower perplexity = more locally grammatical/coherent. Reported either as raw
  perplexity or mapped to a 0–1 coherence score,
  `1 / (1 + max(0, log(ppl) − log(10)))`. This captures the "still makes sense vs.
  word-salad" axis.

The two together separate *interesting loosening* (high drift, coherence
preserved) from *mere degradation* (coherence collapse) — a distinction a single
metric would miss.

## Results

**β produces a real behavioral effect.** Flattening β at early layers (2–3)
collapsed local coherence (perplexity ≈3.7 → ≈19.5) while output stayed
non-degenerate.

**But the entropy signatures don't match EBH:**

- Attention-weight entropy rose at intervened layers (consistent with EBH).
- Spatial-complexity (across-unit activation entropy) was flat (0–0.16%);
  residual-stream LZc slightly *decreased* (≈4–8%).
- Attention-graph degree-distribution entropy (ported from Viol et al. 2017,
  compared at matched edge density to avoid threshold artifacts) *decreased*
  ≈0.51 nats with decreased clustering — the **inverse** of the ayahuasca
  signature. β pushes attention connectivity toward a more regular, lower-entropy
  structure.

## Causal analysis (the main result)

A necessity/sufficiency dissociation shows **entropy is not the cause:**

- **Sufficiency:** perturbing the **value-mixing output** (with attention-weight
  entropy held fixed at baseline) reproduced ~75% of β's effect → re-weighting is
  sufficient.
- **Necessity:** raising **attention-weight entropy** while restoring the output
  toward baseline reproduced ~5% → entropy is not necessary.
- **Direct lever:** directly **raising degree entropy** (independent of β, +0.2
  to +0.43 nats) left reasoning unchanged at every layer (coherence 1.0,
  perplexity ≈baseline). Forcing it downward collapsed generation. The
  EBH-favorable direction produced no loosening.

**Conclusion:** β's effect is carried by structured re-weighting/blending of value
vectors — which value vectors are mixed, in what proportion, into the residual
stream — propagating through depth. Entropy increases are co-occurring readouts
of the same reshaped attention weights, not drivers.

## Interpretation

This recapitulates the EBH/REBUS relationship in an artificial system: entropy and
landscape-flattening act as **signatures** of the perturbed state (REBUS treats
flattening analogically, not mechanistically), while the **causal lever** is
re-weighted signal integration — the general form of REBUS's mechanism (reduced
prior precision in brains; disrupted value blending in the transformer).

## Caveats

- **Mechanism-form, not detail:** β acts bottom-up at early layers; REBUS
  implicates *high-level* prior relaxation. Alignment is "re-weighting as cause,"
  not the full predictive hierarchy.
- **Annealing asymmetry:** annealing raises temperature to *improve* search; β
  degraded output — the operation without the benefit.
- **Scope:** β=0.45, layers 2–3, one 1B model; n=8 (β characterization), n=6
  (degree-entropy intervention). Directional, not powered; metrics are sensitive
  to prompt set and small-n noise.
- **Disanalogy:** brain graph is undirected/resting-state; attention graph is
  directed/causally-masked/task-driven, so the degree-entropy null may reflect
  either no causal role for entropy *or* failure of the analogy to transfer.
- Does not refute EBH as neuroscience (which rests on correlation; this tests
  causation in an artificial system) and does not exhaust the EBH measure family
  (multiscale sample entropy, connectivity-repertoire entropy, meta-state
  complexity untested).

## References

- Carhart-Harris, R. L., & Friston, K. J. (2019). REBUS and the Anarchic Brain:
  Toward a Unified Model of the Brain Action of Psychedelics. *Pharmacological
  Reviews*, 71(3), 316–344.
- Viol, A., Palhano-Fontes, F., Onias, H., de Araujo, D. B., & Viswanathan, G. M.
  (2017). Shannon entropy of brain functional complex networks under the
  influence of the psychedelic Ayahuasca. *Scientific Reports*, 7, 7388.
