---
title: "Language Models for Text Classification: From Bag-of-Words to Jev"
type: summary
source: "raw/language-models-for-text-classification-from-bag-of-words-to-jev.md"
date_ingested: 2026-09-30
tags: [agents]
concepts: []
entities: []
status: draft
related: []
---

# Language Models for Text Classification: From Bag-of-Words to Jev -- Wiki

> Based on Sebastian Raschka's article (September 2026)
> Source: https://magazine.sebastianraschka.com/p/classifier-history-and-jev

---

## Table of Contents

- [Overview](#overview)
- [Classification Before Transformers](#classification-before-transformers)
  - [Representations and Model Trade-offs](#representations-and-model-trade-offs)
  - [IMDb Reference Results](#imdb-reference-results)
- [Transformer Classifiers](#transformer-classifiers)
  - [Encoder, Decoder, and Encoder-Decoder Approaches](#encoder-decoder-and-encoder-decoder-approaches)
  - [Heads Versus Generated Labels](#heads-versus-generated-labels)
- [Jev: General-Purpose Typed Decisions](#jev-general-purpose-typed-decisions)
  - [API Modes](#api-modes)
  - [IMDb Evaluation](#imdb-evaluation)
  - [Where It Fits](#where-it-fits)
- [A Jev-like Model](#a-jev-like-model)
- [Calibration and Training](#calibration-and-training)
  - [Calibration](#calibration)
  - [RLCR as a Related Public Method](#rlcr-as-a-related-public-method)
- [Use Cases and Open Alternatives](#use-cases-and-open-alternatives)
- [Key Takeaways](#key-takeaways)
- [References](#references)

---

## Overview

Text classification has moved from cheap, task-specific **bag-of-words** pipelines to fine-tuned transformers and prompted general LLMs. `Jev` occupies a useful middle ground: a proprietary, low-latency decision model that can apply a typed classification interface to many tasks without per-task fine-tuning.

The important distinction is not that Jev makes classification fundamentally new. Its claimed value is the combination of broad out-of-the-box task coverage, structured probabilities, low cost, and calibrated-confidence-oriented training. For transformer architecture background, see [The Big LLM Architecture Comparison](the-big-llm-architecture-comparison.md) and [A Visual Guide to Attention Variants in Modern LLMs](a-visual-guide-to-attention-variants-in-modern-llms.md).

---

## Classification Before Transformers

### Representations and Model Trade-offs

**Bag-of-words (BoW)** maps a document to a fixed-length, usually sparse vector of vocabulary counts or TF-IDF values. A `50,000`-word vocabulary gives every document a `50,000`-element input regardless of its original length. This is cheap and effective when individual terms strongly predict the class, but word order is lost: “the dog bites the man” and “the man bites the dog” have the same BoW vector. N-grams recover some local order at the cost of a much larger vocabulary.

**Word embeddings** instead represent individual tokens by dense learned vectors. Classic `Word2Vec` and `GloVe` embeddings are context-independent at lookup time, so `bank` has the same initial vector in river-bank and financial-bank contexts. Neural sequence models can then process the embedding sequence and make order matter.

| Approach | Input representation | Captures order/context | Main strength | Main limitation |
|---|---|---|---|---|
| Naive Bayes / logistic regression / SVM / XGBoost | BoW counts or TF-IDF | No; n-grams add local order | Very cheap, strong baseline, easy to inspect | Sparse high-dimensional inputs; weak compositional semantics |
| RNN / LSTM / GRU | Sequential embeddings | Yes, through recurrent state | Variable-length sequence processing | Sequential computation and a fixed-state bottleneck; harder training |
| Text CNN | Sequential embeddings and sliding filters | Local patterns | Parallel over positions; detects useful phrases | Limited receptive field without deeper design |
| Transformer classifier | Token embeddings with attention | Yes, including long-range relations | Benefits from large-scale pretraining | More compute and deployment complexity |

An **RNN** updates a fixed-size hidden state one token at a time. LSTMs (1997) and GRUs (2014) improve retention through gates, but recurrence still constrains parallelism and makes the state an information bottleneck. Text CNNs apply shared filters over adjacent token embeddings; global max pooling makes their output input-length independent, while retaining the strongest detected feature from each channel.

### IMDb Reference Results

The article uses the balanced IMDb sentiment dataset to illustrate that sophistication alone does not guarantee a better result. A small model trained from scratch can overfit, while pretraining or a good linear baseline can be competitive.

| Model / procedure | Test accuracy | Notes |
|---|---:|---|
| BoW + logistic regression | `89.9%` | Cheap reference baseline |
| LSTM trained from scratch | `85.66%` | Reported substantial overfitting |
| Text CNN | `90.07%` | Architecture-dependent result |
| `ULMFiT` (2018) | `95.4%` | Pretrained RNN fine-tuned for IMDb |
| `GPT-2` `124M` fine-tuned classifier | ~`92%` | Decoder model adapted with classification head |
| `ModernBERT` fine-tuned classifier | ~`95%` | Minimal tuning; potentially `1–2%` headroom claimed |

---

## Transformer Classifiers

### Encoder, Decoder, and Encoder-Decoder Approaches

Transformers are normally pretrained before being adapted to a labeled task. The three families differ primarily in what representation is used for the decision and whether prediction is a class-head output or generated text.

| Family | Classification mechanism | Advantages | Caveat |
|---|---|---|---|
| Encoder (`BERT`, `ModernBERT`) | Fine-tune a head over the first `[CLS]` representation | Natural bidirectional classifier; compact and efficient | Needs labeled task data for best results |
| Decoder (`GPT`, `Qwen`) | Prompt for a label, or replace vocabulary head with a classifier | Many current open-weight backbones; can zero/few-shot classify | With causal masking, the decision token must be positioned to attend to all relevant text |
| Encoder-decoder (`T5`) | Add a classification head or generate a label text-to-text | Flexible text-to-text formulation | Usually requires task/domain fine-tuning |

`BERT`-style encoders were built for representations and therefore expose a convenient `[CLS]` token for classification. Decoder-only LLMs can also become classifiers, but a hidden state near the beginning cannot see later tokens under a causal mask; the final non-padding token is a suitable summary position. `T5` uses span-corruption pretraining and can treat classification as generating a target string such as `positive`.

### Heads Versus Generated Labels

**Text-to-text classification** is convenient for a capable prompted LLM, but it spends generative-model capacity producing a constrained answer and can be brittle about output form. A **classification head** returns label scores directly, is cheaper to serve, and gives a clearer probability distribution. The latter is generally preferable when the target schema and domain are known.

---

## Jev: General-Purpose Typed Decisions

`Jev` is TypeSafe AI's proprietary decision model. The underlying architecture, training data, and Reinforcement Learning for Calibrated Decisions (`RLCD`) procedure are not public, so claims about its internals remain inference rather than established fact. The product proposition is a general classifier that approaches an LLM's task flexibility but with classifier-like latency and price.

### API Modes

| API | Output | Best fit | Probability behavior |
|---|---|---|---|
| **Choice** | One named selection plus confidence and per-option probabilities | Mutually exclusive binary or multiclass routing | Candidate probabilities form a distribution |
| **Noul** | `yes`/true probability for a question | Binary classification or independent multi-label decisions | Each question is independent; values need not sum to `1` |
| **Score** | Ordinal rubric level, such as `0`, `1`, or `2` | Ratings and graded assessments | Score semantics are defined by the rubric |

For example, Choice can categorize a support ticket among `billing`, `technical`, and `account`; Noul can separately ask whether a document concerns finance, politics, and technology. The reported Choice response also distinguishes the winning label's probability from a `confidence` field that summarizes concentration of the whole probability distribution.

### IMDb Evaluation

The article's `25,000`-review IMDb test runs put Jev roughly alongside a fine-tuned ModernBERT classifier, without collecting IMDb-specific training data. These results should be treated cautiously: serving is non-deterministic, the exact Jev training set is unknown, and IMDb contamination cannot be excluded.

| Method | Accuracy | Correct / 25,000 | Runtime | Input tokens | Cost |
|---|---:|---:|---:|---:|---:|
| Jev Choice, `jev-1.13.0` | `96.47%` | `24,117` | `22m 24s` | `15,456,663` | `$0.6492` |
| Jev Noul, `jev-1.13.0` | `96.20%` | `24,050` | `23m 03s` | `15,106,663` | `$0.6345` |
| ModernBERT | Similar accuracy | — | `23m` fine-tuning + `7m` evaluation | — | Local hardware dependent |

Repeated Jev calls can differ slightly. A plausible cause is batch-dependent GPU kernel execution changing floating-point operation order, not necessarily intentional sampling.

### Where It Fits

| Situation | Default choice |
|---|---|
| Low-stakes, narrow task with strong lexical clues | BoW + logistic regression baseline |
| Huge volume, stable task, strict privacy or maximal accuracy | Fine-tune and calibrate a specialist classifier |
| One-off or changing decision task | A general decision model such as Jev, or a low-cost LLM |
| Unbounded reasoning/generation is needed too | General LLM |

Jev raises the threshold at which it is worthwhile to build a dedicated model: it may save both an LLM's per-call cost and a team's fine-tuning effort. It does not remove the specialist advantage for high-volume, well-defined tasks.

---

## A Jev-like Model

A similar interface can be placed over `BERT`-, `GPT`-, or `T5`-style backbones, but the interface is not the hard part. To make a flexible Choice API, concatenate the input, task instructions, and each candidate description; use a shared single-scalar scoring head for every candidate; then apply softmax over candidate scores.

For candidate `i`, the head produces `s_i = wᵀh_i + b`. It shares `w` and `b` across all candidates, so the number of options can change without changing the model architecture. For encoders, `h_i` can be the `[CLS]` state; for causal decoders, it is generally the final non-padding token state. Train all scores jointly with cross-entropy against the correct candidate.

This construction makes arbitrary labeled choices possible, but it does not itself yield broad zero-shot generalization. The article's central inference is that Jev's advantage is likely extensive training data, careful evaluation, and calibration work—not a novel one-node head. TypeSafe AI has said its training data are `100%` synthetic; the details of data generation and filtering are undisclosed. More high-quality, representative labeled data can beat marginal hyperparameter tuning: in the author's anecdote, doubling an approximately `300`-sample dataset improved accuracy by more than `10–20%`, versus `2–5%` tuning gains.

---

## Calibration and Training

### Calibration

**Calibration** asks whether reported probabilities match empirical frequencies. If examples assigned about `0.74` positive are actually positive about `74%` of the time, the predictions are calibrated. Classification accuracy alone cannot establish this: predictions of `0.54` and `0.74` choose the same class at a `0.5` threshold but imply very different operational risk.

**Temperature scaling** is a simple post-hoc method: divide fixed model logits by learned positive temperature `T`, then apply softmax. Fit `T` on a held-out calibration set by minimizing cross-entropy. `T > 1` softens overconfident probabilities and `0 < T < 1` sharpens them; because scaling preserves logit order, the argmax class does not change.

Cross-entropy has a calibrated optimum under ideal conditions, and Brier loss shares that optimum. Real neural networks train on finite data and can continue becoming more confident after their probabilities stop generalizing, so calibration must be measured on held-out data rather than assumed.

### RLCR as a Related Public Method

Jev's `RLCD` is proprietary. **Reinforcement Learning with Calibration Rewards (RLCR)** is a related public 2025 method, not evidence of Jev's exact training approach. Unlike binary RL with verifiable rewards (`RLVR`), which gives correct/incorrect outcomes `1`/`0`, RLCR penalizes mismatched confidence:

`R = c - (q - c)^2`

Here `c` is answer correctness (`0` or `1`) and `q` is the model's stated probability of being correct. A wrong answer at `q = 0.9` earns `-0.81`; a wrong answer at `q = 0.2` earns `-0.04`; a correct answer at `q = 0.9` earns `0.99`. The squared term is the Brier penalty.

| Result reported for RLCR | RLVR baseline | RLCR |
|---|---:|---:|
| HotpotQA ECE | `0.37` | `0.03` |
| HotpotQA accuracy | `63.0%` | `62.1%` |
| Six-dataset mean ECE | `0.46` | `0.21` |
| Six-dataset mean accuracy | `53.9%` | `56.2%` |

RLCR has the model generate an answer, uncertainty analysis, and confidence in sequence. A typed-decision model could instead apply a calibration-oriented objective directly to classification probabilities. The article reports only a modest calibration improvement from adding Brier loss to cross-entropy for its ModernBERT experiment, reinforcing that the benefit is empirical rather than guaranteed.

---

## Use Cases and Open Alternatives

Likely Jev-style uses include support-ticket routing, email prioritization, prompt-injection pre-screening, selecting an agent's reasoning effort or `SKILL.md`, judging outputs during evaluation/self-refinement, and retrieving relevant files for agent context. These are especially useful in an agent harness, where a fast decision model can reserve slower LLM calls for substantive reasoning; see [Components of a Coding Agent](components-of-a-coding-agent.md).

Many rapid “Jev clones” pair a fine-tuned `ModernBERT` or `Qwen` model with a similar API. They demonstrate the interface but not the desired generality. `GLiNER` is a longer-standing related open project, though a cited pilot benchmark favored Jev. The article's updated examples report `Contrastive Language Models` at `82.90%` IMDb accuracy and `Laya` at `92.33%`, versus Jev Choice at `96.47%`; the main motivation for a strong open-weight alternative is privacy and local deployment rather than Jev's already low API cost.

The article also notes OpenAI's announced `Decisions API` (limited preview on `2026-09-29`) as a Jev-like service for user-defined finite answers using text or images.

---

## Key Takeaways

1. **Start with a linear baseline.** BoW plus logistic regression remains a cheap, interpretable reference point and can beat poorly trained neural alternatives.

2. **Pretraining matters more than model-family labels.** `ULMFiT`, `ModernBERT`, and adapted decoder models show that pretrained representations can close or surpass older task-specific architectures.

3. **Jev's novelty is product-level generality.** Its differentiated claim is an efficient typed-decision model that works across tasks without one classifier per task, not that classification itself is new.

4. **The structured API encodes real task distinctions.** Use Choice for exclusive labels, Noul for independent labels, and Score for ordered rubrics.

5. **A Jev-shaped head is easy; broad performance is not.** Candidate-wise scalar scoring is simple, whereas data quality, task diversity, calibration, and evaluation determine generalization.

6. **Probabilities need validation.** Accuracy does not make confidence scores trustworthy; evaluate and, when needed, calibrate against held-out data.

7. **Specialists still win in their domain.** For a stable, high-volume task, a fine-tuned, calibrated classifier can remain faster, more private, and more accurate.

---

## References

- Article source: https://magazine.sebastianraschka.com/p/classifier-history-and-jev
- Jev / TypeSafe AI announcement: https://typesafe.ai/blog/introducing-system-one-models-and-jev
- IMDb dataset: https://ai.stanford.edu/~amaas/data/sentiment/
- Bag-of-words tutorial: https://arxiv.org/abs/1410.5329
- Word2Vec: https://arxiv.org/abs/1301.3781
- GloVe: https://aclanthology.org/D14-1162/
- LSTM: https://www.bioinf.jku.at/publications/older/2604.pdf
- GRU: https://aclanthology.org/D14-1179/
- xLSTM: https://proceedings.neurips.cc/paper_files/paper/2024/hash/c2ce2f2701c10a2b2f2ea0bfa43cfaa3-Abstract-Conference.html
- ULMFiT: https://arxiv.org/abs/1801.06146
- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- BERT: https://arxiv.org/abs/1810.04805
- ModernBERT: https://arxiv.org/abs/2412.13663
- T5: https://arxiv.org/abs/1910.10683
- Defeating nondeterminism in LLM inference: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- RLCR: https://arxiv.org/abs/2507.16806
- scikit-learn probability calibration: https://scikit-learn.org/stable/modules/calibration.html
- On Calibration of Modern Neural Networks: https://proceedings.mlr.press/v70/guo17a.html
- Strictly Proper Scoring Rules, Prediction, and Estimation: https://www.tandfonline.com/doi/abs/10.1198/016214506000001437
- GLiNER: https://github.com/urchade/GLiNER
- Jev benchmark report: https://github.com/AbdelStark/jev-benchmarks/blob/main/results/reports/btzsc-pilot-v1.md
- Contrastive Language Models: https://contrastive-lm.notion.site/
- Laya: https://huggingface.co/convaiinnovations/laya
- OpenAI DevDay 2026 recap: https://openai.com/index/devday-2026-recap/
