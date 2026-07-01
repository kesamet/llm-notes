# From GPT-2 to gpt-oss: Analyzing the Architectural Advances -- Wiki

> Based on Sebastian Raschka's article (Aug 2025)
> Source: https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the

---

## Table of Contents

- [Overview](#overview)
- [Evolution from GPT-2](#evolution-from-gpt-2)
- [Removing Dropout](#removing-dropout)
- [RoPE Replaces Absolute Positional Embeddings](#rope-replaces-absolute-positional-embeddings)
- [SwiGLU Replaces GELU](#swiglu-replaces-gelu)
- [Mixture-of-Experts (MoE)](#mixture-of-experts-moe)
- [Grouped Query Attention (GQA)](#grouped-query-attention-gqa)
- [Sliding Window Attention](#sliding-window-attention)
- [RMSNorm Replaces LayerNorm](#rmsnorm-replaces-layernorm)
- [gpt-oss vs. Qwen3 Comparison](#gpt-oss-vs-qwen3-comparison)
- [Width vs. Depth](#width-vs-depth)
- [Few Large vs. Many Small Experts](#few-large-vs-many-small-experts)
- [Attention Bias and Attention Sinks](#attention-bias-and-attention-sinks)
- [Model Specs at a Glance](#model-specs-at-a-glance)
- [Other Noteworthy Details](#other-noteworthy-details)
- [Reasoning Effort Control](#reasoning-effort-control)
- [MXFP4 Quantization](#mxfp4-quantization)
- [Benchmarks](#benchmarks)
- [gpt-oss and GPT-5](#gpt-oss-and-gpt-5)
- [Licensing](#licensing)
- [Key Takeaways](#key-takeaways)
- [References](#references)

---

## Overview

In August 2025, OpenAI released **gpt-oss-20b** and **gpt-oss-120b** -- their first open-weight models since GPT-2 in 2019. Despite six years of progress, the core architecture remains a decoder-only transformer with incremental refinements rather than revolutionary changes. The article traces each architectural change from GPT-2 to gpt-oss, then compares gpt-oss against Qwen3 (a top open-weight competitor) to highlight where OpenAI made distinctive design choices.

The key thesis: most gains in modern LLMs come from **data and training recipes**, not architectural breakthroughs. The transformer remains dominant, and the changes since GPT-2 are a series of sensible efficiency and stability improvements.

---

## Evolution from GPT-2

Both GPT-2 and gpt-oss are decoder-only transformers from the "Attention Is All You Need" lineage. The differences are a set of incremental refinements:

### Removing Dropout

- GPT-2 inherited dropout from the original transformer architecture
- Modern LLMs train for **a single epoch** over massive datasets, so overfitting is not a concern
- A 2025 paper (Pythia 1.4B experiments) confirms dropout **hurts** downstream performance in single-epoch training regimes
- gpt-oss (and virtually all post-GPT-2 LLMs) drops dropout entirely

### RoPE Replaces Absolute Positional Embeddings

| | GPT-2 | gpt-oss |
|---|---|---|
| **Method** | Absolute positional embeddings (learned vector per position, added to token embeddings) | RoPE (rotates Q/K vectors based on position) |
| **Limitation** | Fixed max sequence length; no relative position awareness | Encodes relative position naturally; better length generalization |

- RoPE was introduced in 2021 and became standard with Llama (2023)
- Used by nearly all modern LLMs including gpt-oss and Qwen3

### SwiGLU Replaces GELU

- **GELU** (GPT-2): `0.5x * [1 + erf(x / sqrt(2))]` -- uses expensive Gaussian integral approximation
- **Swish/SiLU** (modern): `x * sigmoid(x)` -- computationally cheaper, similar performance

More importantly, the feed-forward module was upgraded from 2 linear layers to the **GLU (Gated Linear Unit)** variant with 3 linear layers:

| | Standard FFN | SwiGLU |
|---|---|---|
| **Layers** | 2 (fc1, fc2) | 3 (fc1, fc2, fc3) with gating |
| **Parameters** | fc1 + fc2 with full intermediate dim | 3 layers with half intermediate dim each |
| **Net effect** | Baseline | **Fewer total parameters** but better expressivity due to multiplicative gating |

- Example: embedding dim 1024 -> standard FFN = 8.4M params, SwiGLU = 3.1M params
- The gating mechanism adds a multiplicative interaction that improves expressivity (analogous to why deep-and-slim networks outperform shallow-and-wide ones)
- Gemma models are a notable exception, still using GELU

### Mixture-of-Experts (MoE)

gpt-oss replaces the single feed-forward block with **multiple expert feed-forward blocks**, using a router to select a small subset per token:

- Makes the model **sparse** -- large total parameter count, small active parameter count
- Expert weights account for >90% of total model parameters
- Increases model capacity (knowledge absorption) while keeping inference efficient

### Grouped Query Attention (GQA)

| | MHA (GPT-2) | GQA (gpt-oss) |
|---|---|---|
| **Key/Value heads** | One per query head | Shared across groups of query heads |
| **KV cache** | Full size | Reduced (fewer K/V to store and retrieve) |
| **Performance** | Baseline | Comparable per ablation studies (GQA paper, Llama 2 paper) |

### Sliding Window Attention

- Alternates between **full-context GQA** and **sliding-window GQA** (window = 128 tokens) every other layer (1:1 ratio)
- Reduces memory usage and compute per layer
- Gemma 3 uses a 5:1 ratio (more aggressive); Gemma 2 used 1:1 like gpt-oss
- Fun fact: sliding-window attention was already used in GPT-3 (confirmed in the original paper)

### RMSNorm Replaces LayerNorm

| | LayerNorm (GPT-2) | RMSNorm (gpt-oss) |
|---|---|---|
| **Operation** | Subtract mean, divide by std dev (zero mean, unit variance) | Divide by root-mean-square only |
| **Cost** | Two cross-feature reductions (mean + variance) | One reduction (RMS) |
| **Bias term** | Yes | No |

- RMSNorm reduces communication overhead on GPUs, improving training efficiency
- Functionally similar stabilization effect with less compute

---

## gpt-oss vs. Qwen3 Comparison

gpt-oss-20b and Qwen3-30B-A3B are architecturally very similar -- same core components (GQA, SwiGLU, RoPE, MoE, RMSNorm). The differences are in specific design trade-offs:

### Width vs. Depth

| | gpt-oss-20b | Qwen3-30B-A3B |
|---|---|---|
| **Transformer blocks** | 24 | 48 |
| **Embedding dim** | 2880 | 2048 |
| **Expert FFN dim** | 2880 | 768 |
| **Approach** | **Wide** (fewer layers, larger dims) | **Deep** (more layers, smaller dims) |

Trade-offs:
- **Deeper models**: more flexibility, harder to train (gradient instability), sequential layer execution limits parallelism
- **Wider models**: faster inference throughput via better GPU parallelization, higher memory cost
- Gemma 2 ablation (9B params): wider setup scored 52.0 vs. 50.8 for deeper -- slight advantage for width

### Few Large vs. Many Small Experts

| | gpt-oss-20b | Qwen3-30B-A3B |
|---|---|---|
| **Number of experts** | 32 | 128 |
| **Active experts** | 4 | 8 |
| **Expert size** | Large (2880 hidden) | Small (768 hidden) |

- Industry trend favors **many small experts** for better specialization (pioneered by DeepSeek)
- Neither gpt-oss nor Qwen3 uses shared experts (unlike DeepSeek)
- gpt-oss-120b scales up to 32 transformer blocks and more experts while keeping all other dimensions identical -- an unusual scaling strategy (most model families scale proportionally across all dimensions)

### Attention Bias and Attention Sinks

**Attention bias**: gpt-oss reintroduces bias units in attention Q/K/V projections -- a feature last seen in GPT-2 and generally considered redundant. A 2023 paper shows mathematically that key bias is provably redundant, and empirically there is little difference with or without bias.

**Learned attention sinks**: gpt-oss implements attention sinks as **learned per-head bias logits** appended to attention scores (not actual tokens in the input). Purpose:
- Stabilizes attention in long-context scenarios
- Acts as an always-attended anchor that can store generally useful sequence information
- Unlike the original attention sinks paper (which uses real tokens), this approach avoids modifying tokenized inputs

---

## Model Specs at a Glance

| | gpt-oss-20b | gpt-oss-120b | Qwen3-30B-A3B |
|---|---|---|---|
| **Total params** | 20B | 120B | 30B |
| **Active params** | ~1.5B | ~3.6B | 3B |
| **Transformer blocks** | 24 | 32 | 48 |
| **Embedding dim** | 2880 | 2880 | 2048 |
| **Attention heads** | 32 | 32 | 32 |
| **KV heads (GQA)** | 8 | 8 | 4 |
| **Experts** | 32 | 64 | 128 |
| **Active experts** | 4 | 4 | 8 |
| **Expert FFN dim** | 2880 | 2880 | 768 |
| **Sliding window** | Yes (1:1, 128 tokens) | Yes (1:1, 128 tokens) | No |
| **Attention bias** | Yes | Yes | No |
| **Attention sinks** | Yes (learned) | Yes (learned) | No |
| **Shared experts** | No | No | No |
| **License** | Apache 2.0 | Apache 2.0 | Apache 2.0 |

---

## Other Noteworthy Details

### Reasoning Effort Control

gpt-oss models are **reasoning models** (trained with supervised fine-tuning + high-compute RL). They support adjustable reasoning effort via system prompt instructions:

- `Reasoning effort: low` -- short, fast responses
- `Reasoning effort: medium` -- balanced
- `Reasoning effort: high` -- extended reasoning, highest accuracy

This allows balancing cost/compute/accuracy per task. Unlike Qwen3, which initially tried a hybrid thinking toggle but later abandoned it in favor of separate Instruct/Thinking/Coder variants (hybrid mode degraded performance).

OpenAI did **not** release base models (pre-RL), unlike Qwen3 and OLMo -- limiting their usefulness for reasoning research.

### MXFP4 Quantization

OpenAI released gpt-oss with **MXFP4** (microscaling FP4) quantization for MoE experts:

| Model | Without MXFP4 (bf16) | With MXFP4 |
|---|---|---|
| gpt-oss-20b | ~48 GB VRAM | ~16 GB (RTX 50-series+ or via patch on RTX 4090) |
| gpt-oss-120b | ~240 GB VRAM | ~80 GB (single H100) |

- Eliminates need for multi-GPU setups for inference
- The 20b model runs comfortably on a Mac Mini via Ollama (~13.5 GB memory)

### Benchmarks

- At time of writing (Aug 2025), gpt-oss was not yet on the LM Arena leaderboard; Qwen3-Instruct remained the top open-weight model
- On reasoning benchmarks, gpt-oss-120b is on par with OpenAI's proprietary models and Qwen3-235B -- notable given it is roughly half the size
- gpt-oss shows a **higher tendency to hallucinate**, likely due to heavy reasoning-focused training causing "general knowledge forgetting"
- This may matter less as tool-use matures -- models can consult external sources for factual queries, making reasoning capacity more valuable than memorization

### gpt-oss and GPT-5

GPT-5 was released shortly after gpt-oss. The most surprising finding: gpt-oss benchmark scores are remarkably close to GPT-5, suggesting OpenAI's open-weight models are not far behind their best proprietary offerings.

### Licensing

- Both gpt-oss and Qwen3 use **Apache 2.0** -- permissive commercial use, distillation allowed
- gpt-oss is technically **open-weight** (weights + inference code) rather than open-source (no training code or data), despite "oss" standing for "open source software"
- OpenAI correctly describes gpt-oss as open-weight in their announcement

---

## Key Takeaways

1. **Transformers remain dominant** -- No alternative architecture (state-space models, text diffusion) has proven competitive at scale. The highest-ranking non-pure-transformer on LM Arena is a hybrid model.

2. **Incremental refinements add up** -- Every change from GPT-2 (RoPE, SwiGLU, GQA, MoE, RMSNorm, sliding window) is individually modest, but collectively they enable dramatically more efficient and capable models.

3. **Width vs. depth is a real trade-off** -- gpt-oss chose wide (24 layers, 2880 dim) for inference speed; Qwen3 chose deep (48 layers, 2048 dim) for flexibility. Both strategies are viable.

4. **Expert design is still evolving** -- gpt-oss uses few large experts (32); DeepSeek and others use many small experts (128-256). The trend favors many small, but OpenAI's choice may reflect practical scaling concerns.

5. **MXFP4 makes open-weight practical** -- Fitting a 120B model on a single H100 is a meaningful accessibility improvement.

6. **Open-weight models are closing the gap** -- gpt-oss benchmarks are surprisingly close to GPT-5, validating the open-weight approach.

7. **GPT-2 is still the best starting point for learning** -- Its simplicity makes it ideal for understanding transformer fundamentals before layering on modern optimizations.

---

## References

- gpt-oss models: https://huggingface.co/openai/gpt-oss-20b, https://huggingface.co/openai/gpt-oss-120b
- gpt-oss model card: https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf
- gpt-oss announcement: https://openai.com/index/introducing-gpt-oss/
- GPT-2: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- GPT-3: https://arxiv.org/abs/2005.14165
- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- RoPE: https://arxiv.org/abs/2104.09864
- GLU variants: https://arxiv.org/abs/2002.05202
- GQA paper: https://arxiv.org/abs/2305.13245
- LongFormer (SWA): https://arxiv.org/abs/2004.05150
- RMSNorm: https://arxiv.org/abs/1910.07467
- Dropout: https://arxiv.org/abs/1207.0580
- Attention sinks: https://arxiv.org/abs/2309.17453
- Attention bias analysis: https://arxiv.org/abs/2302.08626
- DeepSeekMoE: https://arxiv.org/abs/2401.06066
- DeepSeek V3: https://arxiv.org/abs/2412.19437
- Gemma 2: https://arxiv.org/abs/2408.00118
- Gemma 3: https://arxiv.org/abs/2503.19786
- Llama 2: https://arxiv.org/abs/2307.09288
- Qwen3 from scratch: https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/11_qwen3
- Build a Large Language Model (From Scratch): https://amzn.to/4fqvn0D

