 A Technical Tour of the DeepSeek Models from V3 to V3.2 -- Wiki

> Based on Sebastian Raschka's article (December 2025, updated January 2026)
> Source: https://magazine.sebastianraschka.com/p/technical-deepseek

---

## Table of Contents

- [Overview](#overview)
- [DeepSeek Release Timeline](#deepseek-release-timeline)
- [Hybrid Versus Dedicated Reasoning Models](#hybrid-versus-dedicated-reasoning-models)
- [From DeepSeek V3 to V3.1](#from-deepseek-v3-to-v31)
- [DeepSeek V3.2-Exp and Sparse Attention](#deepseek-v32-exp-and-sparse-attention)
- [DeepSeekMath V2: Self-Verification and Self-Refinement](#deepseekmath-v2-self-verification-and-self-refinement)
- [DeepSeek V3.2](#deepseek-v32)
- [mHC: Manifold-Constrained Hyper-Connections](#mhc-manifold-constrained-hyper-connections)
- [Key Takeaways](#key-takeaways)
- [References](#references)

---

## Overview

This article traces the evolution of DeepSeek's flagship open-weight models from DeepSeek V3 (December 2024) through to DeepSeek V3.2 (December 2025). It covers the architectural innovations -- Multi-Head Latent Attention (MLA), Mixture-of-Experts (MoE), and DeepSeek Sparse Attention (DSA) -- as well as key training advances including Reinforcement Learning with Verifiable Rewards (RLVR), GRPO stability improvements, and the self-verification/self-refinement techniques pioneered in DeepSeekMath V2. DeepSeek V3.2 is notable for achieving GPT-5/Gemini 3.0 Pro-level performance as a fully open-weight model, making these technical details particularly worth studying.

---

## DeepSeek Release Timeline

DeepSeek V3 launched in December 2024 but gained mainstream attention only after DeepSeek R1 (a dedicated reasoning model built on the same architecture) demonstrated strong performance against proprietary models from OpenAI, Google, xAI, and Anthropic.

Key releases in chronological order:

| Release | Date | Role |
|---|---|---|
| DeepSeek V3 | Dec 2024 | Base model with MoE + MLA |
| DeepSeek R1 | Jan 2025 | Dedicated reasoning model (same architecture, RLVR post-training) |
| DeepSeek R1-0528 | May 2025 | Minor version upgrade to R1 with post-training optimizations |
| DeepSeek V3.1 | Mid-2025 | Hybrid reasoning model (instruct + reasoning in one model) |
| DeepSeek V3.2-Exp | Sep 2025 | Experimental release to test sparse attention infrastructure |
| DeepSeekMath V2 | Nov 27, 2025 | Math-focused proof-of-concept for self-verification/self-refinement |
| DeepSeek V3.2 | Dec 1, 2025 | Flagship hybrid model with DSA, self-verification, and GRPO updates |

The V3.2-Exp release was strategically intended to get the ecosystem and inference tooling ready for V3.2, since the new sparse attention mechanism requires custom code.

---

## Hybrid Versus Dedicated Reasoning Models

The LLM landscape has been oscillating between two paradigms for reasoning:

- **Dedicated reasoning models:** Separate models specifically post-trained for reasoning (e.g., DeepSeek R1, early OpenAI o-series).
- **Hybrid models:** A single model that supports both general chat (instruct) and reasoning modes, toggled via prompt template or system prompt (e.g., Qwen3, OpenAI gpt-oss).

| Aspect | Dedicated Reasoning | Hybrid Model |
|---|---|---|
| Example | DeepSeek R1 | DeepSeek V3.1, V3.2 |
| Flexibility | Single mode | User selects mode |
| Development | Easier to optimize for one task | Harder to balance both modes |
| Deployment | Two separate models needed | One model serves all use cases |

DeepSeek's trajectory moved from dedicated reasoning (R1) toward hybrid models (V3.1, V3.2). The author speculates R1 was more of a research testbed for reasoning methods, while V3.2 represents the production-oriented best-overall model. A dedicated R2 may still be in development.

---

## From DeepSeek V3 to V3.1

### Multi-Head Latent Attention (MLA)

Introduced in DeepSeek V2 and carried forward through V3, R1, and all subsequent models, MLA is a memory-saving strategy for KV caching:

1. **Compress** key and value tensors into a lower-dimensional latent space before storing in the KV cache.
2. **Store** the compressed representations (much smaller memory footprint).
3. **Decompress** by projecting back to original size at inference time via an extra matrix multiplication.

This is analogous to the down-projection and up-projection in LoRA. Queries are also compressed, but only during training, not inference.

### Reinforcement Learning with Verifiable Rewards (RLVR)

DeepSeek R1 introduced RLVR to improve reasoning. The core idea: train the model using responses that can be verified symbolically or programmatically (e.g., math solutions checked by a calculator, code verified by a compiler).

The learning algorithm is **GRPO (Group Relative Policy Optimization)**, a simplified variant of PPO that eliminates the critic (value) model. Combined with verifiable rewards, GRPO also removes the reward model, relying entirely on symbolic verification.

| RL Setup | Reward Source | Critic Model |
|---|---|---|
| RLHF with PPO | Reward model (human preferences) | Yes |
| GRPO | Reward model | No |
| RLVR with GRPO | Symbolic verifier (calculator, compiler) | No |

### DeepSeek R1-0528

A minor upgrade with the same architecture. Improvements came from post-training pipeline optimizations and likely more compute at inference time (longer reasoning). No detailed technical report was released.

### DeepSeek V3.1

The first hybrid model in the DeepSeek family, combining instruct and reasoning capabilities in one model. Users switch modes via the chat prompt template. Built on DeepSeek V3.1-Base, which shares the same architecture as V3.

---

## DeepSeek V3.2-Exp and Sparse Attention

The main innovation in V3.2-Exp is **DeepSeek Sparse Attention (DSA)**, which selectively reduces the context tokens each query attends to, improving efficiency especially for long contexts.

### How DSA Works

DSA consists of two components:

1. **Lightning Indexer** -- computes relevance scores between the current query token and all previous tokens using the compressed MLA representations. The score is a weighted sum of ReLU-gated dot products across multiple indexer heads:

`I(t,s) = sum over j of w(t,j) * ReLU(q(t,j) . k(s))`

- `t`: current query position
- `s`: previous token position
- `j`: indexer head index
- `w`: learned per-head weighting coefficients
- ReLU zeroes negative scores; sparsity comes from the token selector, not directly from ReLU

2. **Token Selector** -- keeps only the top-k highest-scoring past tokens (k=2048 by default) and constructs a sparse attention mask that excludes all other tokens.

### DSA vs Other Attention Strategies

| Mechanism | Which tokens are attended | Selection method |
|---|---|---|
| Full causal attention | All previous tokens | None (attend to all) |
| Sliding window attention | Fixed local window | Position-based |
| DeepSeek Sparse Attention | Learned subset of past tokens | Indexer scores + top-k selection |

### Complexity Reduction

DSA reduces attention complexity from **O(L^2)** (quadratic in sequence length) to **O(Lk)** (linear, where k << L is the number of selected tokens).

The goal was not to beat V3.1-Terminus on benchmarks but to minimize performance degradation while gaining significant efficiency improvements.

---

## DeepSeekMath V2: Self-Verification and Self-Refinement

Released November 27, 2025, DeepSeekMath V2 served as a proof-of-concept for techniques later adopted in V3.2. It achieved gold-level scores in math competitions by addressing two shortcomings of standard RLVR:

1. **Correct answers don't guarantee correct reasoning** -- a model can arrive at the right answer through flawed logic.
2. **Some tasks require step-by-step rigor** -- theorem proving needs rigorous derivation, not just a final numerical answer.

### Self-Verification

To address these issues, DeepSeek trained a three-LLM system:

- **LLM 1 (Proof Generator):** Generates mathematical proofs.
- **LLM 2 (Proof Verifier):** Scores proofs on a rubric (1.0 = rigorous, 0.5 = minor errors, 0.0 = fundamentally flawed). Trained via RL with format and score rewards.
- **LLM 3 (Meta-Verifier):** Checks the verifier's assessments to prevent hallucinated criticisms. Improved verifier quality scores from 0.85 to 0.96.

This setup is analogous to GANs: the verifier pushes the generator to produce better proofs, and better proofs in turn push the verifier to be more discerning.

The meta-verifier is used only during training of the verifier and generator -- not at inference time.

### Self-Refinement

Self-refinement lets the model iteratively improve its own answers. A key finding: when a single model both generates and evaluates its own proofs, it tends to claim correctness even when flaws exist. Training under a stronger external verifier resolves this -- the final model internalizes the verification rubrics.

At inference time, a single model performs both generation and verification (no separate verifier LLM needed), because the training process made it strong enough to self-evaluate. Up to 8 refinement iterations were tested, with accuracy still improving and not yet saturating.

---

## DeepSeek V3.2

DeepSeek V3.2 is the culmination of the preceding innovations. It performs at GPT-5 / Gemini 3.0 Pro level and is available as an open-weight model.

### Architecture

Uses **exactly the same architecture as DeepSeek V3.2-Exp**:
- Mixture-of-Experts (MoE)
- Multi-Head Latent Attention (MLA)
- DeepSeek Sparse Attention (DSA)

The interesting developments are in the training methods.

### Reinforcement Learning Updates

DeepSeek V3.2 evolves beyond the pure RLVR approach of R1 into a hybrid reward system:

| Task Domain | Reward Type |
|---|---|
| Reasoning / agentic tasks | Rule-based outcome reward + length penalty + language consistency |
| Math | RLVR + DeepSeekMath V2 self-verification dataset and method |
| General tasks | Generative reward model (LLM-as-a-judge with per-prompt rubrics) |

Notable changes from R1: the format reward was removed, a length penalty was added for agentic tasks, and an LLM-based reward model was introduced for non-verifiable domains.

### GRPO Stability Updates

DeepSeek V3.2 makes targeted improvements to GRPO, staying closer to the original algorithm than alternatives like DAPO or Dr. GRPO:

- **Domain-specific KL strengths:** KL penalty weight tuned per domain (can be zero for math) rather than globally removed.
- **Unbiased KL estimate:** Reweights the KL term with importance ratios so the gradient correctly reflects that samples come from the old policy.
- **Off-policy sequence masking:** Drops rollouts that have both negative advantage and excessive policy drift, preventing learning from stale data.
- **Frozen MoE routing:** Logs which experts were activated during rollout and forces the same routing during training updates.
- **Frozen sampling mask:** Stores and reapplies top-p/top-k selection masks so the training action space matches what was available during sampling.
- **Retains original GRPO normalization:** Unlike Dr. GRPO (which removes length and std normalization) or DAPO (which moves to token-level loss), V3.2 keeps the original normalization and focuses on the fixes above.

### DeepSeek V3.2-Speciale (Extended Thinking)

An extreme reasoning variant trained only on reasoning data during RL, with a reduced length penalty to allow longer outputs. This is a form of inference-time scaling: more tokens generated leads to higher accuracy at greater cost.

---

## mHC: Manifold-Constrained Hyper-Connections

Published December 31, 2025, this DeepSeek research targets an often-overlooked component: the residual path.

| Component | Evolution |
|---|---|
| Normalization | LayerNorm -> RMSNorm -> Dynamic TanH |
| Attention | GQA -> Sliding window -> MLA -> Sparse attention |
| FFN | GeLU -> SiLU -> SwiGLU -> MoE |
| **Residual** | **Identity -> Hyper-Connections -> mHC** |

**Hyper-Connections (HC)** generalize the standard identity residual connection by widening the residual stream into multiple parallel paths with learned mixing.

**mHC** constrains this mixing to lie on a structured, norm-preserving manifold, which significantly improves training stability. The overhead is small, but the gains in convergence and stability are substantial.

---

## Key Takeaways

1. **Architecture continuity.** DeepSeek V3.2 shares the same core architecture (MoE + MLA) as all models since V3. The only architectural addition is DeepSeek Sparse Attention (DSA) from V3.2-Exp.

2. **DSA trades attention scope for efficiency.** By using a learned lightning indexer and top-k token selector instead of attending to all past tokens, DSA reduces attention complexity from O(L^2) to O(Lk), enabling efficient long-context processing.

3. **Hybrid reasoning is the new default.** DeepSeek moved from dedicated reasoning (R1) to hybrid models (V3.1, V3.2) that combine instruct and reasoning modes in a single model, following an industry-wide trend.

4. **Self-verification addresses RLVR's blind spot.** Standard RLVR only checks final answers; the DeepSeekMath V2 approach uses a separately trained verifier (and meta-verifier) to ensure reasoning steps are correct, not just conclusions.

5. **Self-refinement works best with training-time verification.** While a single model handles both generation and verification at inference time, the key is that it was trained under a stronger external verifier, internalizing rigorous evaluation rubrics.

6. **GRPO evolution is conservative but precise.** Rather than adopting aggressive modifications like DAPO or Dr. GRPO, DeepSeek V3.2 makes targeted fixes (unbiased KL, off-policy masking, frozen routing/sampling) while retaining the original GRPO structure.

7. **Reward systems are now hybrid.** V3.2 uses symbolic verifiers for math/code, self-verification for proofs, and LLM-as-a-judge for general tasks -- a pragmatic combination rather than a single approach.

8. **Residual connections are the next frontier.** The mHC research shows that even the residual path, long treated as a simple identity connection, can be improved with learned, norm-preserving manifold constraints.

---

## References

- Sebastian Raschka's article: https://magazine.sebastianraschka.com/p/technical-deepseek
- DeepSeek V3.2 technical report: https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf
- DeepSeek V3.2-Exp model and code: https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp
- DeepSeekMath V2 paper: https://arxiv.org/abs/2511.22570v1
- mHC: Manifold-Constrained Hyper-Connections paper: https://arxiv.org/abs/2512.24880
- Hyper-Connections paper: https://arxiv.org/abs/2409.19606
- The Big LLM Architecture Comparison: https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison
- Understanding Reasoning LLMs: https://magazine.sebastianraschka.com/p/understanding-reasoning-llms
- The State of RL for LLM Reasoning: https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training
- DAPO paper: https://arxiv.org/abs/2503.14476
- Dr. GRPO paper: https://arxiv.org/abs/2503.20783
- DeepSeek V2 paper: https://arxiv.org/abs/2405.04434
- DeepSeek V3/R1 paper: https://arxiv.org/abs/2412.19437

