# A Visual Guide to Attention Variants in Modern LLMs -- Wiki

> Based on Sebastian Raschka's article (March 2026)
> Source: https://magazine.sebastianraschka.com/p/visual-attention-variants

---

## Table of Contents

- [Overview](#overview)
- [Multi-Head Attention (MHA)](#multi-head-attention-mha)
- [Grouped-Query Attention (GQA)](#grouped-query-attention-gqa)
- [Multi-Head Latent Attention (MLA)](#multi-head-latent-attention-mla)
- [Sliding Window Attention (SWA)](#sliding-window-attention-swa)
- [DeepSeek Sparse Attention (DSA)](#deepseek-sparse-attention-dsa)
- [Gated Attention](#gated-attention)
- [Hybrid Attention](#hybrid-attention)
- [Key Takeaways](#key-takeaways)
- [References](#references)

---

## Overview

This article surveys the major attention variants used in prominent open-weight LLM architectures as of early 2026. It traces the evolution from standard Multi-Head Attention (MHA) through efficiency-oriented variants like Grouped-Query Attention (GQA) and Multi-Head Latent Attention (MLA), sparse attention patterns, and the emerging class of hybrid architectures that mix lightweight sequence modules with periodic full-attention layers.

The core tension across all variants is the same: full attention grows quadratically with sequence length, making KV-cache memory the dominant bottleneck at long contexts. Each variant attacks this bottleneck from a different angle — reducing heads (GQA), compressing cached state (MLA), restricting context scope (SWA/DSA), or replacing most attention layers with linear-time modules (hybrids).

---

## Multi-Head Attention (MHA)

MHA is the standard transformer attention mechanism from *Attention Is All You Need*. It runs several self-attention heads in parallel, each with its own learned Q/K/V projections, then concatenates their outputs.

### Why Attention Was Invented

Attention predates transformers. In encoder-decoder RNNs, the encoder compressed an input sentence into a fixed hidden state, creating a bottleneck — the hidden state couldn't store arbitrarily much information. Attention broke this bottleneck by letting the decoder reach back to the full input sequence rather than relying on one compressed state. Transformers kept this core idea but removed recurrence entirely, making attention the primary sequence-processing mechanism.

### Self-Attention Internals

| Component | Role |
|---|---|
| **Q (Query)** | What the current token is looking for |
| **K (Key)** | What each token makes available for matching |
| **V (Value)** | The information mixed into the output once weights are computed |
| **Wq, Wk, Wv** | Learned projection matrices that produce Q, K, V from input embeddings |
| **Attention matrix A** | Emerges from softmax(QK^T / √d); each row is the weight distribution for one token |
| **Causal mask** | Zeros out future positions so each token only attends to prior tokens and itself |

Multi-head attention repeats this mechanism across several heads in parallel. Different heads can specialize — one on local dependencies, another on semantic links, another on syntactic structure.

**Example architectures:** GPT-2, OLMo 2 7B, OLMo 3 7B.

---

## Grouped-Query Attention (GQA)

GQA keeps the full set of query heads but reduces the number of key-value heads, letting multiple query heads share the same K/V projections. This primarily reduces KV-cache memory without drastically changing the decoder recipe.

### Why GQA Became the Default

Standard MHA gives every head its own keys and values — optimal for modeling but expensive once all that state must be stored in the KV cache during inference. GQA lowers both parameter count and KV-cache traffic. It became the de facto replacement for MHA because it is robust, easy to implement, and requires fewer hyperparameter choices than more aggressive alternatives like MLA.

### GQA Is a Spectrum

| Configuration | KV Heads | Cache Cost | Modeling Quality |
|---|---|---|---|
| **MHA** | Equal to query heads | Highest | Best |
| **GQA (typical)** | Small number of groups | Moderate | Near-MHA |
| **Multi-Query Attention (MQA)** | 1 shared group | Lowest | Noticeable degradation |

The sweet spot is usually between MQA and MHA. Cache savings become more pronounced as context length grows.

### GQA in 2026

More advanced variants like MLA are gaining traction, but GQA remains appealing for its simplicity. Recent models like MiniMax M2.5 and Nanbeige 4.1 stayed deliberately classic with GQA alone. Sarvam illustrates the tradeoff directly: its 30B model uses GQA, while its 105B model upgrades to MLA.

**Example architectures (dense):** Llama 3 8B, Qwen3 4B, Gemma 3 27B, Mistral Small 3.1 24B, SmolLM3 3B, Tiny Aya 3.35B.
**Example architectures (MoE):** Llama 4 Maverick, Qwen3 235B-A22B, Step 3.5 Flash 196B, Sarvam 30B.

---

## Multi-Head Latent Attention (MLA)

MLA, introduced in the DeepSeek-V2 paper, reduces KV-cache cost by **compressing what gets stored** rather than reducing how many K/V heads exist. Instead of caching full-resolution key and value tensors, it stores a low-dimensional latent representation and reconstructs the usable state when needed.

### Compression vs. Sharing

| Approach | Strategy | How It Saves Memory |
|---|---|---|
| **GQA** | Sharing | Fewer K/V heads; multiple queries share them |
| **MLA** | Compression | Full heads, but cached as a compressed latent; reconstructed on demand |

### Ablation Results

The DeepSeek-V2 ablation studies showed that GQA can degrade modeling performance below MHA, while MLA held up much better and could even slightly outperform MHA when tuned carefully. This made MLA not just an efficiency move but a quality-preserving one — at least at large scale. Practitioners report that MLA works well at ≥100B parameters; for smaller models, GQA remains easier to tune.

### Adoption After DeepSeek

Once DeepSeek V3/R1 normalized the design, MLA spread to a second wave of architectures: Kimi K2 scaled it up, GLM-5 paired it with DeepSeek Sparse Attention, Ling 2.5 combined it with linear-attention hybrids, and Mistral Large 3 adopted it. The Sarvam 30B vs. 105B pair demonstrates MLA as a concrete architectural upgrade path as model families scale up.

**Example architectures:** DeepSeek V3, Kimi K2, GLM-5, Ling 2.5, Mistral Large 3, Sarvam 105B.

---

## Sliding Window Attention (SWA)

SWA limits each token to attending only to a fixed window of recent tokens instead of the entire prefix. This is often called **local attention**. Most architectures that use SWA combine local-attention layers with periodic global-attention layers so information can still propagate across the full sequence.

### Gemma 3 as Reference

Gemma 2 used a 1:1 local-to-global ratio with a 4096-token window. Gemma 3 pushed to a 5:1 ratio and shrank the window to 1024. The key finding from ablation was that this more aggressive configuration hurt perplexity only slightly.

### Ratio and Window Size

| Model | Local:Global Ratio | Window Size |
|---|---|---|
| **Gemma 3** | 5:1 | 1024 |
| **Xiaomi MiMo-V2-Flash** | 5:1 | 128 |
| **OLMo 3 32B** | 3:1 | — |
| **Arcee Trinity** | 3:1 | — |

SWA is a tunable knob — more aggressive ratios and smaller windows save more memory but risk losing long-range information on local layers.

### Combining SWA with GQA

SWA and GQA address different parts of the inference bottleneck. SWA reduces how much context a local layer considers; GQA reduces how much K/V state each token contributes. Many recent dense models (e.g., Gemma 3) use both.

**Example architectures:** Gemma 3 27B, OLMo 3 32B, Xiaomi MiMo-V2-Flash, Arcee Trinity, Step 3.5 Flash, Tiny Aya.

---

## DeepSeek Sparse Attention (DSA)

DSA appeared in DeepSeek V3.2 and restricts attention to a subset of prior tokens — similar to SWA in spirit but **learned rather than fixed**. Instead of a hard-coded local window, it uses a two-stage mechanism:

1. **Lightning indexer** — computes relevance scores over prior tokens using MLA's compressed representations.
2. **Token selector** — keeps only a top-k subset of high-scoring positions as the sparse attention mask.

### DSA vs. SWA

| Property | SWA | DSA |
|---|---|---|
| Token selection | Fixed local window | Learned, content-dependent |
| Complexity | Simple | Higher (requires indexer + selector) |
| Adoption | Wide | Narrow (DeepSeek V3.2, GLM-5) |

### Pairing with MLA

In DeepSeek V3.2, MLA compresses what gets cached (the representation) and DSA reduces what gets attended to (the pattern). One optimizes the cache, the other optimizes the attention mask on top of it.

**Example architectures:** DeepSeek V3.2, GLM-5.

---

## Gated Attention

Gated attention is not a separate attention family but a set of stability-oriented modifications applied to an otherwise standard scaled-dot-product attention block. It typically appears as the periodic full-attention layer inside hybrid stacks.

### Modifications Over Standard Attention

1. **Output gate** — scales the attention result before adding it back to the residual stream.
2. **Zero-centered QK-Norm** — replaces standard RMSNorm for Q and K projections.
3. **Partial RoPE** — applies rotary position embeddings to only a subset of dimensions.

These are control and stability changes, not architectural redesigns on the scale of MLA or linear attention. In Qwen3-Next and Qwen3.5, gated attention appears as every fourth block, breaking up runs of Gated DeltaNet layers. Arcee Trinity also uses a related gating idea in a non-hybrid stack.

---

## Hybrid Attention

Hybrid attention is a design pattern that replaces most full-attention layers with cheaper linear or state-space sequence modules while retaining a small number of heavier attention layers for precise retrieval. The motivation is long-context efficiency: full attention is quadratic, so at 128k–1M token contexts, using cheaper modules for most layers starts making sense.

### Gated DeltaNet in Qwen3-Next / Qwen3.5

Qwen3-Next introduced a 3:1 mix of **Gated DeltaNet** blocks and **Gated Attention** blocks. Gated DeltaNet is a linear-attention/recurrent-style mechanism that writes to a small fast-weight memory using a delta-rule update, with learned gates controlling how much new information is added vs. how much state is retained. It is closely related to Mamba-2 but uses a DeltaNet-style memory update instead of the Mamba state-space update. Memory growth is much flatter than full attention.

Qwen3.5 promoted this hybrid into the main flagship series, signaling that the approach is considered mature.

### Kimi Linear (Modified Delta Attention)

Kimi Linear keeps the 3:1 pattern but changes both sides:
- **Lightweight side:** Kimi Delta Attention uses channel-wise gating (finer control) instead of Gated DeltaNet's scalar gate per head.
- **Heavy side:** Gated MLA layers replace gated attention.

### Ling 2.5 (Lightning Attention)

Ling 2.5 swaps in **Lightning Attention** (a simpler recurrent linear attention variant) on the lightweight side and keeps **MLA** on the heavy side. Reported substantially faster than Kimi K2 at 32k tokens.

### Nemotron (Mamba-2 Hybrid)

Nemotron pushes furthest from the transformer baseline. Nemotron 3 Nano interleaves Mamba-2 state-space blocks with sparse MoE layers, using self-attention only in a small subset of layers. Nemotron 3 Super adds latent MoE and shared-weight multi-token prediction on top.

| Architecture | Lightweight Module | Heavy Module | Ratio |
|---|---|---|---|
| **Qwen3-Next / Qwen3.5** | Gated DeltaNet | Gated Attention | 3:1 |
| **Kimi Linear** | Kimi Delta Attention | Gated MLA | 3:1 |
| **Ling 2.5** | Lightning Attention | MLA | — |
| **Nemotron 3 Nano** | Mamba-2 | Self-Attention | Mostly Mamba-2 |

---

## Key Takeaways

1. **GQA is the pragmatic default.** It reduces KV-cache cost by sharing K/V heads across query groups — simple to implement, well-understood, and still used in most new dense and MoE models.

2. **MLA trades implementation complexity for quality-preserving compression.** By caching a low-dimensional latent instead of full K/V tensors, MLA matches or exceeds MHA quality while rivaling GQA's memory savings — but it works best at ≥100B scale.

3. **SWA is a tunable knob, not a binary choice.** The local-to-global ratio and window size can be adjusted; Gemma 3 showed that aggressive settings (5:1, 1024 window) barely hurt perplexity.

4. **DeepSeek Sparse Attention learns which tokens matter.** Unlike SWA's fixed window, DSA uses a learned indexer-selector to pick which prior tokens to attend to — more powerful but harder to implement and not yet widely adopted.

5. **Hybrid architectures are production-ready.** Qwen3.5 promoting the Gated DeltaNet hybrid into its main flagship line signals that mixing linear-time modules with periodic full attention is no longer experimental.

6. **Different variants attack different parts of the same bottleneck.** GQA/MLA compress the cache representation; SWA/DSA restrict the attention pattern; hybrids replace the attention mechanism itself in most layers. These are complementary and often combined.

7. **Inference tooling still favors classic designs.** Despite theoretical advantages, hybrid architectures have less optimized inference stacks — classic GQA setups often deliver better tok/sec when running locally.

8. **Watch for Mamba-3 and attention residuals.** The author flags Mamba-3 layers replacing Gated DeltaNet in hybrids and attention residuals as the next developments to track.

---

## References

- LLM Architecture Gallery: https://sebastianraschka.com/llm-architecture-gallery/
- Understanding and Coding Self-Attention: https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention
- The Big LLM Architecture Comparison: https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison
- Coding the KV Cache in LLMs: https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms
- A Dream of Spring for Open-Weight LLMs: https://magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight
- From DeepSeek V3 to V3.2: https://magazine.sebastianraschka.com/p/technical-deepseek
- LLMs-from-scratch (code): https://github.com/rasbt/LLMs-from-scratch
- GQA paper: https://arxiv.org/abs/2305.13245
- DeepSeek-V2 paper (MLA): https://arxiv.org/abs/2405.04434
- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- Gemma 3 paper: https://arxiv.org/abs/2503.19786
- DeepSeek V3.2 paper: https://arxiv.org/abs/2512.02556
- Gated Attention paper: https://arxiv.org/abs/2505.06708
- Gated Delta Networks paper: https://arxiv.org/abs/2412.06464
- Mamba-3 paper: https://arxiv.org/abs/2603.15569
- Attention Residuals paper: https://arxiv.org/abs/2603.15031

