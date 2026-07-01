# Understanding the 4 Main Approaches to LLM Evaluation -- Wiki

> Based on Sebastian Raschka's article (Oct 2025)
> Source: https://magazine.sebastianraschka.com/p/llm-evaluation-4-approaches

---

## Table of Contents

- [Overview](#overview)
- [Evaluation Taxonomy](#evaluation-taxonomy)
- [Method 1: Multiple-Choice Benchmarks](#method-1-multiple-choice-benchmarks)
- [How It Works](#how-it-works)
- [Scoring Variants](#scoring-variants)
- [Strengths and Limitations](#strengths-and-limitations)
- [Method 2: Verification-Based Evaluation](#method-2-verification-based-evaluation)
- [How It Works](#how-it-works-1)
- [Strengths and Limitations](#strengths-and-limitations-1)
- [Method 3: Arena-Style Leaderboards](#method-3-arena-style-leaderboards)
- [How It Works](#how-it-works-2)
- [Elo Rating System](#elo-rating-system)
- [Bradley-Terry Model](#bradley-terry-model)
- [Strengths and Limitations](#strengths-and-limitations-2)
- [Method 4: LLM-as-a-Judge](#method-4-llm-as-a-judge)
- [How It Works](#how-it-works-3)
- [Rubric Design](#rubric-design)
- [Process Reward Models](#process-reward-models)
- [Strengths and Limitations](#strengths-and-limitations-3)
- [Comparison of All Four Methods](#comparison-of-all-four-methods)
- [Practical Guidance](#practical-guidance)
- [Key Takeaways](#key-takeaways)
- [References](#references)

---

## Overview

LLM evaluation falls into four main approaches: **multiple-choice benchmarks**, **verifiers**, **arena-style leaderboards**, and **LLM-as-a-judge**. These divide into two groups -- benchmark-based evaluation (multiple choice, verifiers) and judgment-based evaluation (leaderboards, LLM judges). No single method captures the full picture; robust evaluation combines several.

Other internal metrics like training loss, perplexity, and reward scores exist but are typically used during model development rather than for model comparison.

---

## Evaluation Taxonomy

| Group | Method | What It Measures | Answer Format |
|---|---|---|---|
| **Benchmark-based** | Multiple-choice | Knowledge recall | Constrained (A/B/C/D) |
| **Benchmark-based** | Verifiers | Correctness (math, code) | Free-form with extracted answer |
| **Judgment-based** | Leaderboards | Human preference | Free-form |
| **Judgment-based** | LLM-as-a-judge | Quality per rubric | Free-form |

---

## Method 1: Multiple-Choice Benchmarks

### How It Works

The model is presented with a question and predefined answer options (e.g., A/B/C/D). Its selected answer is compared against the ground truth. Performance is reported as **accuracy** (fraction correct).

The canonical example is **MMLU** (Massive Multitask Language Understanding) -- 57 subjects, ~16,000 questions spanning high school math to biology. Random guessing yields 25% accuracy.

Few-shot variants (e.g., 5-shot MMLU) prepend solved examples to the prompt to demonstrate the expected response format, though modern base models typically don't require this.

### Scoring Variants

Three scoring methods are commonly used:

| Variant | Mechanism | Notes |
|---|---|---|
| **Letter matching** | Generate tokens, extract first A/B/C/D letter | Simplest; depends on generation behavior |
| **Log-probability (per-choice)** | Compare log-prob of each answer token given the prompt | More robust; doesn't require generation |
| **Log-probability (full answer)** | Compare log-prob of the complete correct answer as continuation | Used for reasoning model evaluation |

Regardless of variant, the evaluation reduces to checking whether the model selects the correct predefined option.

### Strengths and Limitations

A high MMLU score does not guarantee real-world utility, but a low score reliably flags knowledge gaps. Multiple-choice benchmarks are most useful as a **sanity check** -- for example, verifying that a fine-tuned model hasn't forgotten base knowledge.

---

## Method 2: Verification-Based Evaluation

### How It Works

The model generates a **free-form answer** (potentially including intermediate reasoning steps). The final answer is extracted -- often from a `\boxed{}` format -- and compared to the ground truth using a **verifier**. Verifiers can be symbolic (exact match, calculator), code-execution-based (run the solution and check output), or tool-augmented.

This approach is the cornerstone of **reasoning model evaluation and development** because:
- Math problem variations can be generated programmatically at unlimited scale
- Step-by-step reasoning benefits from free-form generation
- Ground truth is deterministic and unambiguous

### Strengths and Limitations

Verifiers provide objective, reproducible grading for domains with clear correct answers. The main constraint is that they are limited to **verifiable domains** (math, code, formal logic). Outcome-only verifiers evaluate just the final answer, not reasoning quality. Building robust verifiers can also introduce engineering complexity.

---

## Method 3: Arena-Style Leaderboards

### How It Works

Users submit prompts to two models (often anonymously selected), view both responses side-by-side, and vote for the one they prefer. Pairwise preferences are aggregated into a ranking. **LM Arena** (formerly Chatbot Arena) is the most prominent example.

This is the only evaluation method that directly answers: "Which model do people actually prefer on real prompts?" It implicitly captures style, helpfulness, safety, and other hard-to-formalize qualities.

### Elo Rating System

The original LM Arena ranking used the **Elo rating system** (from chess). Each model starts at a baseline rating (e.g., 1000). After each pairwise comparison, ratings update based on how surprising the outcome was:

- **Expected match** (similar ratings): small update (~16 points each way)
- **Upset** (low-rated model wins): large update (~32 points each way)
- **Dominant win** (high-rated model wins): minimal update (~0.3 points)

The update formula for the winner's expected score:

```
expected_winner = 1 / (1 + 10^((rating_loser - rating_winner) / 400))
```

Elo is **order-sensitive** -- the same set of votes processed in different order can produce slightly different final scores. Shuffling and averaging across multiple runs mitigates this.

### Bradley-Terry Model

LM Arena has since transitioned to the **Bradley-Terry model**, which estimates all ratings jointly via a statistical fit over the entire dataset. Key advantages over Elo:

- **Order-invariant**: results don't depend on the sequence votes are processed
- **Confidence intervals**: provides uncertainty estimates for rankings
- Scores are calibrated to the familiar Elo scale for continuity

Despite the switch, the term "Elo" remains widely used in the LLM community when discussing model rankings.

### Strengths and Limitations

Leaderboards capture real-world preference but are expensive to run, subject to voting biases (style over correctness, user demographics, prompt selection), and do not provide instant feedback during active model development.

---

## Method 4: LLM-as-a-Judge

### How It Works

A separate, typically stronger LLM evaluates the target model's response against a reference answer using a predefined **rubric** (grading guide). The judge outputs a score (e.g., 1-5) and often a justification.

Common judge setups use leading proprietary models via API (e.g., GPT-5), though open-weight models like `gpt-oss:20b` (see the [From GPT-2 to gpt-oss](from-gpt-2-to-gpt-oss-analyzing-the-architectural-advances.md) page for architectural details) work well as local judges via tools like Ollama.

A key insight: **evaluating an answer is generally easier than generating one**, which is why smaller or weaker models can still serve as effective judges.

### Rubric Design

A typical 5-point rubric:

| Score | Criterion |
|---|---|
| 1 | Irrelevant, incorrect, or excessively verbose |
| 2 | Partially addresses instruction; major errors or omissions |
| 3 | Somewhat addresses instruction; incomplete or unclear |
| 4 | Mostly correct; minor errors or lack of clarity |
| 5 | Fully correct, clear, accurate, and concise |

The rubric is included in the judge's prompt alongside the instruction, the reference answer, and the candidate answer. Rubric choice significantly influences results -- different rubrics can yield different rankings for the same set of models.

### Process Reward Models

**Process Reward Models (PRMs)** are a related class of learned models that provide step-by-step reward signals during reinforcement learning training. They differ from standard judges:

| | LLM Judge | PRM |
|---|---|---|
| **Evaluates** | Final answer (+ optionally reasoning) | Each intermediate reasoning step |
| **Primary use** | Post-hoc evaluation | Training signal for RL |
| **Scalability** | High (general-purpose LLM) | Hard to train reliably at scale |

PRMs are best categorized as "step-level judges" primarily for training, not pure evaluation. Notably, DeepSeek R1 did not adopt PRMs and instead relied on verifiers for reasoning training.

### Strengths and Limitations

LLM judges scale well and don't require human voters, but they share biases with humans: sensitivity to answer style, prompt design, and the specific judge model chosen. They lack the reproducibility of fixed benchmarks. Using ensembles of judge models can improve robustness.

---

## Comparison of All Four Methods

| Dimension | Multiple Choice | Verifiers | Leaderboards | LLM Judge |
|---|---|---|---|---|
| **Answer format** | Constrained | Free-form | Free-form | Free-form |
| **Metric** | Accuracy | Accuracy | Elo / BT score | Rubric score |
| **Cost** | Low | Low-Medium | High (human time) | Medium (API cost) |
| **Reproducibility** | High | High | Low (population effects) | Medium (model/rubric dependent) |
| **Domain scope** | General knowledge | Verifiable (math, code) | Any | Any |
| **Measures correctness** | Yes | Yes | Indirectly | Partially (rubric dependent) |
| **Measures style/helpfulness** | No | No | Yes | Partially |
| **Real-time dev feedback** | Yes | Yes | No | Yes |
| **Gameable** | Yes (data contamination) | Less so | Yes (style hacking) | Yes (judge hacking) |

---

## Practical Guidance

The best evaluation strategy combines multiple methods. A useful mental model:

1. **Strong multiple-choice score** -- the model has solid general knowledge
2. **Strong verifier score** -- the model answers technical questions correctly
3. **Weak LLM-judge and leaderboard scores** -- the model may struggle to articulate responses well; could benefit from RLHF

For domain-specific applications (e.g., legal, medical), standard benchmarks like MMLU serve as a sanity check, but evaluation should ultimately be **tailored to the target domain using proprietary data**. This also mitigates the risk of data contamination (the model having seen test data during training).

---

## Key Takeaways

1. **No single evaluation method is sufficient.** Each approach has blind spots -- multiple-choice misses free-form ability, verifiers only work in deterministic domains, leaderboards are expensive and biased, judges depend on rubric and model choice. Combine at least two or three.

2. **Multiple-choice benchmarks are a floor, not a ceiling.** A high MMLU score doesn't prove real-world utility, but a low score reliably indicates knowledge gaps. Use them as diagnostic sanity checks, not as the primary measure of model quality.

3. **Verification-based evaluation is the backbone of reasoning model development.** Its combination of deterministic grading and unlimited synthetic data generation makes it the most practical approach for iterating on reasoning capabilities.

4. **Leaderboards measure what people actually care about -- but imprecisely.** LM Arena remains the closest proxy for real-world preference, though it conflates style preference with correctness and is expensive to scale.

5. **LLM-as-a-judge is the most flexible approach.** It scales like a benchmark but captures open-ended quality like a leaderboard. The main risk is that judge biases become evaluation biases. Ensembles and careful rubric design help.

6. **Always evaluate on domain-specific data.** Generic benchmarks tell you whether a model is broadly capable, but only testing on your target domain with proprietary data confirms it will work for your use case -- and ensures the model hasn't memorized the test set.

7. **The Bradley-Terry model is a strict upgrade over Elo for leaderboard ranking.** It's order-invariant and provides confidence intervals. Despite this, the community still uses "Elo" colloquially.

---

## References

- MMLU dataset: https://huggingface.co/datasets/cais/mmlu
- LM Arena (formerly Chatbot Arena): https://lmarena.ai/
- Elo rating system: https://en.wikipedia.org/wiki/Elo_rating_system
- Bradley-Terry model: LM Arena blog
- Phudge (specialized judge model): https://arxiv.org/abs/2405.08029
- MATH-500 evaluation code: https://github.com/rasbt/reasoning-from-scratch/tree/main/chF/04_llm-judge
- MMLU scoring variants code: https://github.com/rasbt/reasoning-from-scratch/tree/main/chF/02_mmlu
- Leaderboard implementation code: https://github.com/rasbt/reasoning-from-scratch/tree/main/chF/03_leaderboards
- Build a Reasoning Model (From Scratch): https://mng.bz/Nwr7
- Process Reward Models: discussed in DeepSeek R1 (https://arxiv.org/abs/2501.12948)

