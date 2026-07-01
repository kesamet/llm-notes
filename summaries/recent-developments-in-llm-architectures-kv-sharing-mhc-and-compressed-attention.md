# Recent Developments in LLM Architectures: KV Sharing, mHC, and Compressed Attention -- Wiki

> Based on Sebastian Raschka's article (May 2026)
> Source: https://magazine.sebastianraschka.com/p/recent-developments-in-llm-architectures

---

## Table of Contents

- [Overview](#overview)
- [1. Cross-Layer KV Sharing (Gemma 4 E2B/E4B)](#1-cross-layer-kv-sharing-gemma-4-e2be4b)
- [2. Per-Layer Embeddings and Effective Size (Gemma 4 E2B/E4B)](#2-per-layer-embeddings-and-effective-size-gemma-4-e2be4b)
- [3. Layer-Wise Attention Budgeting (Laguna XS.2)](#3-layer-wise-attention-budgeting-laguna-xs2)
- [4. Compressed Convolutional Attention (ZAYA1-8B)](#4-compressed-convolutional-attention-zaya1-8b)
- [5. DeepSeek V4: mHC and Compressed Attention](#5-deepseek-v4-mhc-and-compressed-attention)
- [5.1 Manifold-Constrained Hyper-Connections (mHC)](#51-manifold-constrained-hyper-connections-mhc)
- [5.2 CSA and HCA](#52-csa-and-hca)
- [Architecture Comparison](#architecture-comparison)
- [Key Takeaways](#key-takeaways)
- [References](#references)

---

## Overview

This article covers four architecture advances from the April-May 2026 open-weight LLM releases, all motivated by the same theme: **reducing long-context inference cost** without shrinking total model capacity. As reasoning models and agent workflows keep more tokens around, KV-cache size, memory traffic, and attention cost become the dominant constraints.

The techniques covered are:

| Model | Technique | What It Reduces |
|---|---|---|
| Gemma 4 E2B/E4B | Cross-layer KV sharing | KV cache memory |
| Gemma 4 E2B/E4B | Per-Layer Embeddings (PLE) | Compute per token (while preserving capacity) |
| Laguna XS.2 | Per-layer query-head budgeting | Attention compute on expensive layers |
| ZAYA1-8B | Compressed Convolutional Attention (CCA) | KV cache + attention FLOPs |
| DeepSeek V4 | mHC + CSA/HCA | Residual expressiveness + long-context KV/FLOPs |

For background on MHA, GQA, MLA, SWA, and DSA, see the [Visual Guide to Attention Variants](a-visual-guide-to-attention-variants-in-modern-llms.md) wiki page. For MoE, normalization, and broader architecture context, see [The Big LLM Architecture Comparison](the-big-llm-architecture-comparison.md).

---

## 1. Cross-Layer KV Sharing (Gemma 4 E2B/E4B)

### Concept

Standard GQA shares K/V heads across query heads within a single layer. **Cross-layer KV sharing** (also called cross-layer attention) goes further: later layers reuse the K/V tensors computed by an earlier layer of the same attention type, instead of computing their own K/V projections. Each layer still computes its own Q projection, so attention patterns remain layer-specific.

This technique was proposed in Brandon et al., "Reducing Transformer Key-Value Cache Size with Cross-Layer Attention" (NeurIPS 2024), but Gemma 4 is the first widely deployed model to adopt it.

### How It Works in Gemma 4

- Sliding-window layers share KV with the most recent earlier sliding-window layer.
- Full-attention layers share KV with the most recent earlier full-attention layer.
- Only the first N layers compute their own KV; remaining layers reuse.

| Model | Total Layers | KV-Computing Layers | KV-Sharing Layers |
|---|---|---|---|
| Gemma 4 E2B | 35 | 15 | 20 |
| Gemma 4 E4B | 42 | 24 | 18 |

### Memory Savings

Since roughly half the layers share KV, the technique saves approximately half the KV cache. Concrete numbers at bfloat16 with 128K context:

| Model | KV Cache Savings |
|---|---|
| E2B | ~2.7 GB |
| E4B | ~6 GB |

### Tradeoff

KV-sharing reduces model capacity (later layers lose per-layer KV specialization). The cross-layer attention paper reports minimal impact for small models, but no public ablation exists for larger architectures.

---

## 2. Per-Layer Embeddings and Effective Size (Gemma 4 E2B/E4B)

### The "Effective" Naming

The "E" in E2B and E4B stands for "effective". The model reports two parameter counts:

| Model | Effective Params (transformer stack) | Total Params (with embeddings) |
|---|---|---|
| E2B | 2.3B | 5.1B |
| E4B | 4.5B | 8B |

The main transformer compute operates at the smaller "effective" size, while additional capacity lives in per-layer embedding tables.

### How PLE Works

1. **PLE construction** (computed once, outside the transformer blocks):
- Token IDs go through a per-layer embedding lookup.
- Normal token embeddings go through a linear projection into the same space.
- These two contributions are added, scaled, and reshaped into one slice per layer.

2. **Inside each transformer block** (layer l):
- Attention and FFN residual updates proceed normally.
- The resulting hidden state **z** gates the layer-specific PLE vector (ple_l).
- The gated PLE is projected to model hidden size, normalized, and added as an extra residual update.

### Why PLE Instead of a Bigger Model

PLE stores capacity in cheap lookup-style embedding tables rather than in wider attention/FFN matrices. This provides more token-specific information per layer without scaling the expensive compute path. It is most beneficial for small models where the transformer stack cannot otherwise store enough capacity. Larger models already get this benefit from MoE routing.

### Open Questions

No public comparison exists between E2B (2.3B effective + PLE) versus a plain 2.3B or plain 5.1B Gemma model. The design's empirical value relative to simply scaling up remains Google's internal finding.

---

## 3. Layer-Wise Attention Budgeting (Laguna XS.2)

### Model Context

**Laguna XS.2** is the first open-weight model from Poolside, a Europe-based company focused on coding LLMs. The model has 40 layers using a familiar sliding-window + global attention mix (30 SWA layers, 10 global layers, 512-token window).

### Per-Layer Query-Head Counts

The novel element is **varying the number of query heads per layer** via a `num_attention_heads_per_layer` config setting:

| Layer Type | Query Heads per KV Head | KV Heads |
|---|---|---|
| Sliding-window layers | 8 | 8 |
| Global (full) attention layers | 6 | 8 |

Global layers are more expensive because they attend over the full context, so Laguna gives them fewer query heads to reduce their cost. Sliding-window layers are cheap (only 512-token context), so they can afford more query heads.

### Precedent

The broader idea of varying capacity by layer traces to Apple's OpenELM (2024). Laguna XS.2 is the most prominent production-style open model to apply per-layer query-head budgeting specifically.

---

## 4. Compressed Convolutional Attention (ZAYA1-8B)

### Model Context

**ZAYA1-8B** is developed by Zyphra and is notable for being trained on AMD GPUs. It uses a sparse MoE setup (1 routed expert active per token) with 40 transformer blocks combining CCA/GQA attention and MoE FFN layers.

### CCA vs. MLA

Both CCA and MLA introduce a compressed latent representation into the attention block, but they use it differently:

| Property | MLA | CCA |
|---|---|---|
| What is compressed | K, V (for caching) | Q, K, V (all three) |
| Where attention happens | Full-dimensional space (after decompression) | Compressed latent space (before decompression) |
| What it reduces | KV cache size | KV cache + attention FLOPs (prefill and training) |
| Up-projection timing | Before attention (decompress K/V for scoring) | After attention (decompress the output) |

CCA's key advantage: since attention is computed in the compressed space, it reduces not only cache memory but also the computational cost of the attention operation itself.

### Convolutional Mixing

Compression makes Q, K, V narrower, which can reduce expressiveness. CCA compensates by applying **convolutional mixing** on compressed Q and K before computing attention scores. This gives compressed representations local context cheaply. V is not convolved because it represents content to be averaged, not the scoring mechanism.

Two types of mixing are applied:
- **Sequence mixing**: 1D convolution along the token dimension on compressed Q/K.
- **Channel mixing**: Similar convolution along the hidden dimension.

### Performance Claims

The standalone CCA paper reports CCA outperforming MLA under comparable compression settings in their experiments. However, ZAYA1-8B is a single-model data point; broader adoption is needed to confirm generality.

### Reference

CCA was introduced in a standalone paper (October 2025) before being deployed in ZAYA1-8B: "Compressed Convolutional Attention: Efficient Attention in a Compressed Latent Space" (arXiv:2510.04476).

---

## 5. DeepSeek V4: mHC and Compressed Attention

DeepSeek V4 is the largest release of early 2026. V4-Pro is the most parameter-sparse MoE among major models by active-parameter share. The two main architecture advances over DeepSeek V3/V3.2 are:

1. **mHC** -- widens the residual pathway
2. **CSA/HCA** -- compresses attention along the sequence dimension

For background on DeepSeek V3/V3.2's MLA and DSA, see the [DeepSeek Models Technical Tour](a-technical-tour-of-the-deepseek-models-from-v3-to-v32.md) wiki page, which also introduces the mHC concept from the December 2025 paper. DeepSeek V4 is the first production deployment of mHC.

### 5.1 Manifold-Constrained Hyper-Connections (mHC)

The existing [DeepSeek wiki page](a-technical-tour-of-the-deepseek-models-from-v3-to-v32.md) covers the mHC concept. Key new information from the V4 deployment:

**What mHC does**: Replaces the single residual stream with **n parallel residual streams** (n=4 in DeepSeek V4) connected by learned mappings. The attention/MoE layers still operate on normal hidden size via Pre/Post Mappings that combine and redistribute streams.

**HC vs. mHC constraint**:

| Aspect | Hyper-Connections (HC) | mHC |
|---|---|---|
| Residual mapping | Unconstrained learned matrix | Projected onto doubly stochastic manifold (rows/columns sum to 1) |
| Pre/Post mapping | Unconstrained | Non-negative, bounded (prevents cancellation) |
| Stability at depth | Can amplify/shrink signals | Stable redistribution |
| Purpose | More expressive residual | Same expressiveness + safe scaling |

**Overhead**: The mHC paper reported only 6.7% additional training time for n=4 streams in a 27B experiment. FLOPs are nearly unchanged (the mappings operate over the small n-axis, not the large hidden dimension). Practical overhead comes from memory traffic and implementation complexity more than arithmetic.

**V4 deployment**: DeepSeek V4 uses mHC throughout all transformer blocks, confirming the technique scales to flagship production models.

### 5.2 CSA and HCA

CSA (Compressed Sparse Attention) and HCA (Heavily Compressed Attention) are DeepSeek V4's attention mechanisms. They are fundamentally different from MLA:

| Dimension | MLA (V3/V3.2) | CSA/HCA (V4) |
|---|---|---|
| Compression axis | Per-token representation (narrower KV entries) | Sequence length (fewer KV entries) |
| Cache entries | One latent per token | Groups of tokens compressed into fewer entries |
| What's sacrificed | Some per-token fidelity | Some token-level granularity |

#### CSA (Compressed Sparse Attention)

- Groups every **m=4 tokens** into one compressed KV entry (mild compression).
- Uses a **DSA-style selector** (learned sparse top-k) to pick which compressed entries to attend to.
- Also includes a 128-token **sliding-window branch** for recent uncompressed tokens.

#### HCA (Heavily Compressed Attention)

- Groups every **m'=128 tokens** into one compressed KV entry (aggressive compression).
- Uses **dense attention** over the heavily compressed entries (affordable because there are so few).
- Also includes a 128-token sliding-window branch.

#### Why Both?

CSA and HCA are complementary: CSA retains more detail but must be sparse, HCA retains less detail but can afford dense coverage. DeepSeek V4 **interleaves CSA and HCA layers** to get both fine-grained selection and broad global coverage.

#### Efficiency Gains at 1M Context

| Metric (vs. DeepSeek V3.2) | V4-Pro | V4-Flash |
|---|---|---|
| Single-token inference FLOPs | 27% | 10% |
| KV cache size | 10% | 7% |

These are dramatic reductions: V4-Flash uses only 7% of V3.2's KV cache at 1M tokens.

#### Caveats

CSA/HCA is not necessarily "better" than MLA in general. It is a more aggressive long-context design with more complexity. No ablation study isolates CSA/HCA's contribution from the full V4 recipe (better data, Muon optimizer, mHC, precision optimizations, etc.).

---

## Architecture Comparison

| Model | Release | Attention | KV Reduction Strategy | Other Notable Feature |
|---|---|---|---|---|
| Gemma 4 E2B | Apr 2026 | MQA + SWA (4:1) | Cross-layer KV sharing (~50% cache savings) | Per-Layer Embeddings (PLE) |
| Gemma 4 E4B | Apr 2026 | GQA + SWA | Cross-layer KV sharing (~50% cache savings) | Per-Layer Embeddings (PLE) |
| Gemma 4 26B MoE | Apr 2026 | GQA + SWA (5:1) | Standard GQA | V=K in global layers |
| Gemma 4 31B Dense | Apr 2026 | GQA + SWA (5:1) | Standard GQA | p-RoPE (25%) |
| Laguna XS.2 | Apr 2026 | GQA + SWA (30:10) | Per-layer query-head budgeting | Per-head attention-output gating |
| ZAYA1-8B | May 2026 | CCA + GQA (4:1) | Attention in compressed latent space | 1 active expert per token; trained on AMD |
| DeepSeek V4-Pro | May 2026 | CSA + HCA (interleaved) | Sequence-dimension compression (m=4, m'=128) | mHC (n=4 parallel residual streams) |
| DeepSeek V4-Flash | May 2026 | CSA + HCA (interleaved) | Sequence-dimension compression | mHC; 7% of V3.2 KV at 1M tokens |

---

## Key Takeaways

1. **KV-cache reduction is the dominant theme.** Every architecture in this article attacks KV-cache cost from a different angle -- sharing across layers (Gemma 4), operating in compressed space (CCA), or compressing along the sequence dimension (CSA/HCA).

2. **Cross-layer KV sharing is simple and effective.** Gemma 4's approach of reusing K/V tensors from earlier layers saves ~50% of KV cache with minimal reported quality loss. Expect wider adoption in small/edge models.

3. **CCA moves attention into compressed space.** Unlike MLA which decompresses before attention, CCA performs the attention operation in the compressed latent space itself, reducing both cache and compute. The convolutional mixing compensates for lost expressiveness.

4. **DeepSeek V4's CSA/HCA compresses the sequence dimension.** This is a different axis from MLA's per-token compression. By grouping tokens into compressed entries, V4 achieves 90-93% KV cache reduction at 1M context relative to V3.2.

5. **mHC is now production-validated.** First tested in a 27B experiment (Dec 2025), DeepSeek V4 confirms that manifold-constrained parallel residual streams scale to flagship models with minimal overhead.

6. **Per-layer attention budgeting is an overlooked knob.** Laguna XS.2 shows that varying query heads per layer (more for cheap local layers, fewer for expensive global layers) is a straightforward way to redistribute compute.

7. **Complexity is increasing.** The original transformer block was 50-100 lines of PyTorch. These attention variants likely 10x that complexity. The basic decoder recipe persists, but individual components are becoming highly specialized.

8. **Architecture vs. training remains asymmetric.** These techniques reduce runtime cost, but qualitative modeling performance is still largely driven by data quality, quantity, and training recipes (as evidenced by Gemma 4's quality leap with near-identical architecture to Gemma 3).

---

## References

- Sebastian Raschka, "Recent Developments in LLM Architectures": https://magazine.sebastianraschka.com/p/recent-developments-in-llm-architectures
- LLM Architecture Gallery: https://sebastianraschka.com/llm-architecture-gallery/
- Cross-Layer Attention paper (Brandon et al., NeurIPS 2024): https://arxiv.org/abs/2405.12981
- Gemma 4 from-scratch implementation: https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/17_gemma4
- OpenELM (Apple, 2024): https://arxiv.org/abs/2404.14619
- CCA paper: https://arxiv.org/abs/2510.04476
- ZAYA1-8B technical report: https://arxiv.org/abs/2605.05365
- mHC paper: https://arxiv.org/abs/2512.24880
- Hyper-Connections paper: https://arxiv.org/abs/2409.19606
- DeepSeek V4 paper: (referenced via article; check DeepSeek's official channels for latest link)
- Laguna XS.2 config: https://huggingface.co/poolside/Laguna-XS.2/blob/main/config.json
- A Visual Guide to Attention Variants: https://magazine.sebastianraschka.com/p/visual-attention-variants
- The Big LLM Architecture Comparison: https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison
- Coding the KV Cache in LLMs: https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms
- DeepSeek V3.2 write-up: https://magazine.sebastianraschka.com/p/technical-deepseek
- Build a Reasoning Model (From Scratch): https://mng.bz/Nwr7
