# The Big LLM Architecture Comparison -- Wiki

> Based on Sebastian Raschka's article (Jul 2025, last updated Apr 2026)
> Source: https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

---

## Table of Contents

- [Overview](#overview)
- [Key Architectural Concepts](#key-architectural-concepts)
- [Attention Mechanisms](#attention-mechanisms)
- [Mixture-of-Experts (MoE)](#mixture-of-experts-moe)
- [Normalization Strategies](#normalization-strategies)
- [Positional Encoding](#positional-encoding)
- [Multi-Token Prediction (MTP)](#multi-token-prediction-mtp)
- [Model Architectures](#model-architectures)
- [1. DeepSeek V3/R1](#1-deepseek-v3r1)
- [2. OLMo 2](#2-olmo-2)
- [3. Gemma 3](#3-gemma-3)
- [4. Mistral Small 3.1](#4-mistral-small-31)
- [5. Llama 4](#5-llama-4)
- [6. Qwen3](#6-qwen3)
- [7. SmolLM3](#7-smollm3)
- [8. Kimi K2](#8-kimi-k2)
- [9. GPT-OSS](#9-gpt-oss)
- [10. Grok 2.5](#10-grok-25)
- [11. GLM-4.5](#11-glm-45)
- [12. Qwen3-Next](#12-qwen3-next)
- [13. MiniMax-M2](#13-minimax-m2)
- [14. Kimi Linear](#14-kimi-linear)
- [15. Olmo 3](#15-olmo-3)
- [16. DeepSeek V3.2](#16-deepseek-v32)
- [17. Mistral 3](#17-mistral-3)
- [18. Nemotron 3](#18-nemotron-3)
- [19. Xiaomi MiMo-V2-Flash](#19-xiaomi-mimo-v2-flash)
- [20. Arcee AI Trinity Large](#20-arcee-ai-trinity-large)
- [21. GLM-5](#21-glm-5)
- [22. Gemma 4](#22-gemma-4)
- [Architecture Comparison Table](#architecture-comparison-table)
- [Trends and Takeaways](#trends-and-takeaways)
- [References](#references)

---

## Overview

Seven years after the original GPT architecture, modern LLMs remain structurally similar at their core -- decoder-only transformers with incremental refinements. The main evolutionary changes include:

- **Positional embeddings**: absolute -> RoPE -> partial RoPE / NoPE
- **Attention**: Multi-Head Attention (MHA) -> Grouped-Query Attention (GQA) -> Multi-Head Latent Attention (MLA) -> linear attention hybrids
- **Activation functions**: GELU -> SwiGLU
- **Scaling strategy**: dense models -> Mixture-of-Experts (MoE)

Despite these being "minor" refinements, they collectively enable orders-of-magnitude improvements in efficiency and capability.

---

## Key Architectural Concepts

### Attention Mechanisms

| Mechanism | How It Works | KV Cache Savings | Used By |
|---|---|---|---|
| **Multi-Head Attention (MHA)** | Each head has its own Q, K, V projections | None (baseline) | OLMo 2, Olmo 3 7B |
| **Grouped-Query Attention (GQA)** | Multiple query heads share K/V projections; reduces KV head count | Moderate | Llama 4, Gemma 3/4, Qwen3, Mistral, SmolLM3, MiniMax-M2 |
| **Multi-Head Latent Attention (MLA)** | Compresses K/V into lower-dimensional latent space before caching; decompresses at inference | High | DeepSeek V3/R1/V3.2, Kimi K2, Mistral 3 Large, GLM-5 |
| **Sliding Window Attention (SWA)** | Restricts attention to a local window around each token (local); combined with periodic global layers | High (smaller KV cache) | Gemma 3/4 (5:1), Olmo 3, gpt-oss (1:1), Xiaomi MiMo (5:1), Trinity Large (3:1) |
| **Gated DeltaNet** | Linear attention with gated fast-weight memory update; O(n) complexity | Very high | Qwen3-Next, Kimi Linear |
| **Kimi Delta Attention (KDA)** | Refinement of Gated DeltaNet with channel-wise gating (instead of scalar) | Very high | Kimi Linear |
| **Mamba-2** | State-space model with gated hidden state; linear complexity | Very high | Nemotron 3 Nano/Super |

**Sliding window ratios** (local:global):
- Gemma 3/4, Xiaomi MiMo: **5:1** (5 local layers per 1 global)
- gpt-oss: **1:1** (every other layer)
- Olmo 3, Trinity Large: **3:1**
- Qwen3-Next, Kimi Linear: **3:1** (DeltaNet:full attention)

### Mixture-of-Experts (MoE)

Replaces the single FeedForward block per transformer layer with multiple expert FeedForward blocks. A router selects a small subset of experts per token, making the model **sparse** -- large total parameter count but small active parameter count.

| Model | Total Params | Active Params | Experts | Active Experts | Shared Expert |
|---|---|---|---|---|---|
| DeepSeek V3 | 671B | 37B | 256 | 8 + 1 shared | Yes |
| Llama 4 Maverick | 400B | 17B | 128 | 2 | No |
| Qwen3 235B-A22B | 235B | 22B | 128 | 8 | No |
| Kimi K2 | 1T | 32B | 384 | 8 + 1 shared | Yes |
| gpt-oss 120B | 120B | ~3.6B | 32 | 4 | No |
| GLM-4.5 | 355B | 32B | 160 | 8 + 1 shared | Yes |
| GLM-5 | 744B | 40B | 256 | 8 + 1 shared | Yes |
| Mistral 3 Large | 675B | 41B | 128 | 8 + 1 shared | Yes |
| MiniMax-M2 | 230B | ~10B | 128 | 8 | No |
| Xiaomi MiMo-V2 | 309B | 15B | -- | -- | -- |
| Nemotron 3 Nano | 30B | 3B | 128 | 6 + 1 shared | Yes |
| Trinity Large | 400B | 13B | -- | -- | -- |
| Gemma 4 MoE | 27B | 4B | -- | -- | -- |

**Key design choices:**
- **Shared experts**: Always-on expert processing every token; reduces redundancy across routed experts. Used by DeepSeek, GLM, Kimi K2, Mistral 3, Nemotron, Qwen3-Next. Notably *not* used by Qwen3, MiniMax-M2, or gpt-oss.
- **Many small vs. few large experts**: DeepSeek pioneered many small experts (256 x 2048 hidden). gpt-oss and Grok 2.5 use fewer, larger experts. Trend favors many small experts for better specialization.
- **Dense initial layers**: DeepSeek V3, GLM-4.5/5, and others keep the first 3 layers dense (no MoE) for training stability.

### Normalization Strategies

| Strategy | Placement | Used By |
|---|---|---|
| **Pre-Norm** | RMSNorm *before* attention and FFN (inside residual) | GPT-2, Llama 3, Qwen3, most modern LLMs |
| **Post-Norm (OLMo flavor)** | RMSNorm *after* attention and FFN (inside residual) | OLMo 2, Olmo 3 |
| **Pre + Post-Norm** | RMSNorm both before *and* after each sub-layer | Gemma 2/3/4 |
| **Depth-scaled sandwich norm** | Pre + Post, with post-norm gain initialized to 1/sqrt(L) | Trinity Large |
| **QK-Norm** | RMSNorm applied to queries and keys inside attention, before RoPE | OLMo 2, Gemma 2/3, Qwen3, many others |
| **Per-layer QK-Norm** | Separate QK-Norm parameters per attention head (not shared) | MiniMax-M2 |

**Why it matters**: Normalization placement primarily affects training stability. Post-norm (OLMo style) helps prevent loss spikes. Pre+Post (Gemma style) gets best of both worlds with minimal overhead since RMSNorm is cheap.

### Positional Encoding

| Method | Description | Used By |
|---|---|---|
| **RoPE** | Rotates Q/K vectors relative to token position | Most modern LLMs |
| **Partial RoPE** | RoPE applied to only a subset of head dimensions | MiniMax-M1/M2 (50%), Gemma 4 (25%) |
| **NoPE** | No explicit positional encoding; relies on causal mask for order | SmolLM3 (every 4th layer), Trinity Large (global layers), Kimi Linear (MLA layers) |
| **YaRN** | RoPE rescaling for context length extension | Qwen3, Olmo 3 |

**NoPE insight**: The causal attention mask already provides implicit directional information. NoPE has been shown to improve length generalization -- model performance degrades less with longer sequences.

### Multi-Token Prediction (MTP)

Trains the LLM to predict multiple future tokens (t+1 ... t+k) at each position, rather than just the next token. Provides richer training signal and can enable speculative decoding at inference.

- **Training only**: DeepSeek V3/V3.2, GLM-4.5
- **Training + inference (speculative decoding)**: Qwen3-Next, Nemotron 3 Super, Xiaomi MiMo-V2

---

## Model Architectures

### 1. DeepSeek V3/R1
- **Released**: Dec 2024 (V3) / Jan 2025 (R1)
- **Size**: 671B total, 37B active
- **Key innovations**:
- **Multi-Head Latent Attention (MLA)**: Compresses K/V into lower-dimensional latent space. Outperforms both MHA and GQA in ablation studies from the DeepSeek-V2 paper.
- **MoE with shared expert**: 256 experts, 8 routed + 1 shared active per token. Shared expert handles common patterns, freeing routed experts for specialization.
- **Impact**: Established the architectural template that Kimi K2, Mistral 3 Large, and GLM-5 would later adopt.

### 2. OLMo 2
- **Released**: Jan 2025
- **Org**: Allen Institute for AI
- **Key innovations**:
- **Post-Norm (inside residual)**: RMSNorm placed *after* attention and FFN, but still within skip connections. Improves training stability.
- **QK-Norm**: RMSNorm on queries and keys before RoPE. Combined with post-norm, eliminates training loss spikes.
- **Notable**: Still uses traditional MHA (no GQA). Valued for full transparency (code, data, training details).

### 3. Gemma 3
- **Released**: Mar 2025
- **Org**: Google
- **Key innovations**:
- **Sliding window attention** in a 5:1 ratio (5 local : 1 global). Window size reduced from 4096 (Gemma 2) to 1024. Minimal impact on perplexity per ablation studies.
- **Pre + Post-Norm**: RMSNorm both before and after attention/FFN modules. Extra normalization has negligible compute cost.
- **Also**: Large 256k vocabulary for multilingual support; 27B sweet spot size.
- **Variant**: Gemma 3n -- optimized for phones using Per-Layer Embedding (PLE) and MatFormer slicing.

### 4. Mistral Small 3.1
- **Released**: Mar 2025
- **Size**: 24B dense
- **Key choices**: Abandoned sliding window attention (was in earlier Mistral models). Uses standard GQA. Faster inference than Gemma 3 27B, likely due to avoiding SWA overhead and using optimized FlashAttention.

### 5. Llama 4
- **Released**: Apr 2025
- **Org**: Meta
- **Size**: Maverick -- 400B total, 17B active
- **Key choices**: Adopted MoE (first for Llama series). Uses GQA (not MLA). Fewer but larger experts (2 active with 8192 hidden) vs. DeepSeek's many small experts. Alternates MoE and dense layers.

### 6. Qwen3
- **Released**: May 2025
- **Org**: Alibaba
- **Dense variants**: 0.6B, 1.7B, 4B, 8B, 14B, 32B
- **MoE variants**: 30B-A3B, 235B-A22B
- **Key choices**: Standard GQA + SwiGLU + RoPE architecture. Notably dropped shared experts (Qwen2.5-MoE had them). Qwen3 0.6B is a great small model -- deeper (more layers) but narrower than Llama 3 1B.
- **Dense vs MoE**: Dense for simplicity and fine-tuning; MoE for efficient inference at scale.

### 7. SmolLM3
- **Released**: Jun 2025
- **Org**: Hugging Face
- **Size**: 3B dense
- **Key innovation**:
- **NoPE (No Positional Embedding)**: Omits RoPE in every 4th layer. Causal mask provides implicit ordering. Improves length generalization per the NoPE paper.

### 8. Kimi K2
- **Released**: Jun 2025
- **Org**: Moonshot AI
- **Size**: 1T total, 32B active -- possibly the largest open-weight LLM of this generation
- **Architecture**: DeepSeek V3 architecture with more experts (384) and fewer MLA heads. First production model to use the Muon optimizer (instead of AdamW) at this scale.
- **Variant**: Kimi K2 Thinking (Nov 2025) -- same architecture, 256k context (up from 128k).

### 9. GPT-OSS
- **Released**: Aug 2025
- **Org**: OpenAI (first open-weight since GPT-2)
- **Sizes**: 20B, 120B (MoE)
- **Key design choices**:
- **Wide over deep**: 24 layers with 2880 embedding dim (vs. Qwen3's 48 layers with 2048 dim). Wider = faster inference via parallelization.
- **Few large experts**: 32 experts (vs. typical 128+), 4 active.
- **Attention bias**: Reintroduced bias units in attention (absent since GPT-2 era). Empirically shown to have minimal impact.
- **Learned attention sinks**: Per-head bias logits appended to attention scores; stabilizes long-context attention without modifying input tokens.
- **Sliding window attention**: Every other layer (1:1 ratio).

### 10. Grok 2.5
- **Released**: Aug 2025
- **Org**: xAI
- **Size**: 270B MoE
- **Key choices**: Few large experts (8). Uses a shared SwiGLU module functioning as shared expert (with doubled intermediate dim). Rare look at a real production system released as open weights.

### 11. GLM-4.5
- **Released**: Summer 2025
- **Org**: z.AI (Zhipu AI)
- **Sizes**: 355B flagship (32B active), 106B Air variant
- **Key choices**: DeepSeek-style MoE with shared expert and 3 dense initial layers. Retains attention bias (like GPT-2 and gpt-oss). Instruction/reasoning hybrid optimized for function calling and agentic tasks.

### 12. Qwen3-Next
- **Released**: Sep 2025
- **Size**: 80B-A3B
- **Key innovations**:
- **Gated DeltaNet + Gated Attention hybrid** (3:1 ratio): 3 linear-attention DeltaNet blocks per 1 full attention block. Enables 262k native context.
- **More experts + shared expert**: 4x more experts than Qwen3 235B, reintroduced shared expert.
- **Multi-Token Prediction**: Used at both training and inference (speculative decoding).
- **Variant**: Qwen3-Coder-Next (Feb 2026) -- same architecture, fine-tuned for code. Outperforms much larger models on coding benchmarks.

### 13. MiniMax-M2
- **Released**: Oct 2025
- **Size**: 230B total, ~10B active
- **Key choices**: Went *back* to full attention (M1 used lightning/linear attention -- didn't work well for reasoning and multi-turn). Per-layer QK-Norm (unique parameters per head). Very sparse: only 4.37% of parameters active per step. No shared expert.
- **Lesson**: Linear attention is tricky in production for reasoning and multi-turn tasks.

### 14. Kimi Linear
- **Released**: Oct 2025
- **Org**: Moonshot AI
- **Size**: 48B
- **Key innovations**:
- **Kimi Delta Attention (KDA)**: Refinement of Gated DeltaNet with channel-wise (instead of scalar) gating for better long-context reasoning.
- **MLA for global layers**: Uses MLA (from DeepSeek) with NoPE instead of GQA with RoPE.
- **3:1 hybrid ratio**: Same as Qwen3-Next but with KDA instead of plain DeltaNet.
- **Result**: As fast as DeltaNet, much faster than MLA, with higher benchmark performance than both.

### 15. Olmo 3
- **Released**: Nov 2025
- **Org**: Allen Institute for AI
- **Sizes**: 7B, 32B (base, instruct, reasoning variants)
- **Key choices**: Retains OLMo 2's post-norm for stability. 7B uses MHA; 32B uses GQA. Both use sliding window attention. YaRN for context extension (64k) on global layers only.

### 16. DeepSeek V3.2
- **Released**: Dec 2025
- **Architecture**: DeepSeek V3 + sparse attention mechanism for improved efficiency. On par with GPT-5.1 and Gemini 3.0 Pro on certain benchmarks.

### 17. Mistral 3
- **Released**: Dec 2025
- **Size**: 675B MoE (41B active) flagship + 3B/8B/14B dense Ministral models
- **Architecture**: Effectively the same as DeepSeek V3 (MLA + MoE with shared expert) -- doubled expert size, halved expert count. Optimized for NVIDIA Blackwell chips.
- **Note**: Second model series (after Kimi K2) to adopt the DeepSeek V3 architecture wholesale.

### 18. Nemotron 3
- **Released**: Dec 2025 (Nano), Mar 2026 (Super)
- **Org**: NVIDIA
- **Nano (30B-A3B)**: Mamba-2 + Transformer hybrid. 52 layers organized as 13 macro blocks. Only a few GQA layers; most layers are Mamba-2 + MoE. 128 experts, 6 + 1 shared active.
- **Super (120B-A12B)**: Adds MTP for speculative decoding and **latent experts** (down-project 4096->1024 before expert processing). 2x faster than Qwen3.5 122B at similar quality.
- **Significance**: Most aggressive use of non-attention layers among major models.

### 19. Xiaomi MiMo-V2-Flash
- **Released**: Dec 2025
- **Size**: 309B MoE, 15B active
- **Key choices**: SWA with 5:1 ratio and aggressive window size of only 128 tokens (8x smaller than Gemma 3). Multi-token prediction. Matches DeepSeek V3.2 quality at half the parameters.

### 20. Arcee AI Trinity Large
- **Released**: Jan 2026
- **Size**: 400B MoE (13B active)
- **Key innovations**:
- **Gated attention**: Elementwise sigmoid gate on attention output (reduces attention sinks, improves long-sequence generalization).
- **Depth-scaled sandwich norm**: Pre+Post RMSNorm where post-norm gain is initialized to 1/sqrt(L).
- **NoPE in global layers** + QK-Norm.
- SWA with 3:1 ratio, 4096 window.

### 21. GLM-5
- **Released**: Feb 2026
- **Org**: z.AI (Zhipu AI)
- **Size**: 744B total, 40B active
- **Architecture**: Adopts DeepSeek's MLA and sparse attention (from V3.2). 256 experts (up from 160 in GLM-4.5). Fewer transformer blocks (78 vs. 92) to reduce inference latency.
- **Benchmarks**: On par with GPT-5.2, Gemini Pro 3, and Claude 4.6 Opus.

### 22. Gemma 4
- **Released**: Apr 2026
- **Org**: Google
- **Size**: 31B dense + 27B MoE (26B-A4B)
- **Key changes from Gemma 3**:
- Reuses keys as values (V=K) in global attention layers for further KV cache reduction.
- **Partial RoPE (p-RoPE)**: Only 25% of frequency pairs get positional information.
- **Result**: Huge performance leap over Gemma 3 despite minimal architectural changes -- underscores that training recipes matter as much as architecture.

---

## Architecture Comparison Table

| Model | Type | Total / Active Params | Attention | MoE | Norm | Pos Encoding | Context |
|---|---|---|---|---|---|---|---|
| DeepSeek V3 | MoE | 671B / 37B | MLA | 256E, 8+1 active | Pre-Norm | RoPE | 128k |
| OLMo 2 | Dense | 7B/13B | MHA | -- | Post-Norm + QK-Norm | RoPE | -- |
| Gemma 3 | Dense | 27B | GQA + SWA (5:1) | -- | Pre+Post-Norm + QK-Norm | RoPE | 128k |
| Mistral Small 3.1 | Dense | 24B | GQA | -- | Pre-Norm | RoPE | 128k |
| Llama 4 Maverick | MoE | 400B / 17B | GQA | 128E, 2 active | Pre-Norm | RoPE | 128k |
| Qwen3 | Dense+MoE | 0.6B-235B | GQA + QK-Norm | 128E, 8 active (MoE) | Pre-Norm | RoPE | 32k-131k |
| SmolLM3 | Dense | 3B | GQA | -- | Pre-Norm | RoPE + NoPE (1:4) | -- |
| Kimi K2 | MoE | 1T / 32B | MLA | 384E, 8+1 active | Pre-Norm | RoPE | 128k-256k |
| gpt-oss | MoE | 20B/120B | GQA + SWA (1:1) + sinks | 32E, 4 active | Pre-Norm | RoPE | -- |
| Grok 2.5 | MoE | 270B | GQA | 8E (large) + shared SwiGLU | Pre-Norm | RoPE | -- |
| GLM-4.5 | MoE | 355B / 32B | GQA + attn bias | 160E, 8+1 active | Pre-Norm | RoPE | -- |
| Qwen3-Next | MoE | 80B / 3B | DeltaNet + Gated Attn (3:1) | 512E, 8+1 active | Pre-Norm | Partial RoPE | 262k |
| MiniMax-M2 | MoE | 230B / ~10B | GQA | 128E, 8 active | Pre-Norm + per-layer QK-Norm | Partial RoPE | -- |
| Kimi Linear | Hybrid | 48B | KDA + MLA (3:1) | -- | Pre-Norm | NoPE (MLA) | -- |
| Olmo 3 | Dense | 7B / 32B | MHA/GQA + SWA | -- | Post-Norm + QK-Norm | RoPE + YaRN | 64k |
| DeepSeek V3.2 | MoE | ~671B / 37B | MLA + sparse attn | 256E, 8+1 active | Pre-Norm | RoPE | -- |
| Mistral 3 Large | MoE | 675B / 41B | MLA | 128E, 8+1 active | Pre-Norm | RoPE | -- |
| Nemotron 3 Nano | Hybrid | 30B / 3B | Mamba-2 + GQA (few) | 128E, 6+1 active | Pre-Norm | RoPE | -- |
| Nemotron 3 Super | Hybrid | 120B / 12B | Mamba-2 + GQA + MTP | Latent MoE | Pre-Norm | RoPE | -- |
| Xiaomi MiMo-V2 | MoE | 309B / 15B | GQA + SWA (5:1, w=128) + MTP | MoE | Pre-Norm | RoPE | -- |
| Trinity Large | MoE | 400B / 13B | Gated GQA + SWA (3:1) | MoE (coarse) | Depth-scaled sandwich | RoPE + NoPE (global) | -- |
| GLM-5 | MoE | 744B / 40B | MLA + sparse attn | 256E, 8+1 active | Pre-Norm | RoPE | -- |
| Gemma 4 | Dense+MoE | 31B / 27B MoE | GQA + SWA (5:1), V=K | -- / MoE | Pre+Post-Norm | p-RoPE (25%) | -- |

---

## Trends and Takeaways

### 1. MoE Is the New Default for Large Models
Nearly every model above 100B uses MoE. The trend is toward **more, smaller experts** (DeepSeek's 256 x 2048) rather than fewer large ones, with shared experts becoming common for handling universal patterns.

### 2. DeepSeek V3 as a Reference Architecture
Three separate model families (Kimi K2, Mistral 3 Large, GLM-5) adopted DeepSeek V3's architecture nearly wholesale. Its combination of MLA + MoE with shared experts has become a proven template.

### 3. Attention Efficiency Is the Main Battleground
The 2025-2026 landscape shows multiple approaches competing to reduce attention's quadratic cost:
- **GQA** (reduce KV heads) -- most popular
- **MLA** (compress KV cache) -- best quality/efficiency trade-off
- **Sliding window** (limit context per layer) -- easy to implement, surprisingly effective
- **Linear attention** (Gated DeltaNet, Mamba) -- O(n), promising but tricky for reasoning

### 4. Linear Attention: Promising but Not Yet Proven at Scale
MiniMax went from linear attention (M1) back to full attention (M2) because linear attention struggled with reasoning and multi-turn tasks. However, Kimi Linear and Qwen3-Next show that hybrid approaches (3:1 DeltaNet:full attention) can work well at smaller scales.

### 5. Normalization Keeps Evolving
- Pre-Norm remains the default
- Post-Norm (OLMo style) helps training stability
- Pre+Post-Norm (Gemma) is gaining adoption
- QK-Norm is now nearly universal

### 6. Architecture Matters Less Than Training
Gemma 4's huge quality leap over Gemma 3 with near-identical architecture demonstrates that training data, recipes, and post-training matter enormously. Architecture is necessary but not sufficient.

### 7. Width vs. Depth Trade-off
- Deeper models (more layers) are more flexible but harder to train and slower (layers can't be parallelized)
- Wider models (larger embedding/hidden dims) are faster at inference
- gpt-oss chose width; Qwen3 chose depth; both are viable

---

## References

- DeepSeek V3: https://arxiv.org/abs/2412.19437
- DeepSeek R1: https://arxiv.org/abs/2501.12948
- DeepSeek V2 (MLA origin): https://arxiv.org/abs/2405.04434
- OLMo 2: https://arxiv.org/abs/2501.00656
- Gemma 3: https://arxiv.org/abs/2503.19786
- Llama 4: https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- Qwen3-Next: https://qwen.ai/blog
- SmolLM3: https://huggingface.co/blog/smollm3
- Kimi K2: https://moonshotai.github.io/Kimi-K2/
- Kimi Linear: https://arxiv.org/abs/2510.26692
- gpt-oss: https://openai.com/index/introducing-gpt-oss/
- GLM-4.5: https://arxiv.org/abs/2508.06471
- GLM-5: https://z.ai/blog/glm-5
- MiniMax-M2: https://huggingface.co/MiniMaxAI/MiniMax-M2
- Nemotron 3 Nano: https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf
- Nemotron 3 Super: https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf
- Trinity Large: https://github.com/arcee-ai/trinity-large-tech-report
- GQA paper: https://arxiv.org/abs/2305.13245
- NoPE paper: https://arxiv.org/abs/2305.19466
- Gated DeltaNet: https://arxiv.org/abs/2412.06464
- Multi-Token Prediction: https://arxiv.org/abs/2404.19737
- Attention sinks: https://arxiv.org/abs/2309.17453
- LongFormer (SWA): https://arxiv.org/abs/2004.05150
- Mamba-2: https://arxiv.org/abs/2405.21060
- YaRN: https://arxiv.org/abs/2309.00071
- LLM Architecture Gallery: https://sebastianraschka.com/llm-architecture-gallery/

