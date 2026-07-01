# Beyond Standard LLMs -- Wiki

> Based on Sebastian Raschka's article (November 2025)
> Source: https://magazine.sebastianraschka.com/p/beyond-standard-llms

---

## Table of Contents

- [Overview](#overview)
- [Transformer-Based LLMs: The Baseline](#transformer-based-llms-the-baseline)
- [Linear Attention Hybrids](#linear-attention-hybrids)
- [Text Diffusion Models](#text-diffusion-models)
- [Code World Models](#code-world-models)
- [Small Recursive Transformers](#small-recursive-transformers)
- [Comparative Summary](#comparative-summary)
- [Key Takeaways](#key-takeaways)
- [References](#references)

---

## Overview

While autoregressive decoder-style transformers (DeepSeek R1, Llama 4, Qwen3, etc.) remain the state-of-the-art for text and code, a growing number of alternative architectures are emerging that challenge different aspects of the standard paradigm. This article surveys four such alternatives: **linear attention hybrids** that reduce the quadratic cost of attention, **text diffusion models** that generate tokens in parallel rather than sequentially, **code world models** that learn to simulate program execution to improve coding performance, and **small recursive transformers** that achieve strong reasoning through iterative self-refinement with tiny parameter counts. Each approach targets a different axis -- efficiency, parallelism, modeling depth, or compactness -- and none yet fully replaces autoregressive transformers, but together they sketch the frontier of where LLM architecture research is headed.

---

## Transformer-Based LLMs: The Baseline

Standard autoregressive transformers built on the *Attention Is All You Need* architecture remain the dominant paradigm. Notable open-weight models from late 2024 through 2025 include DeepSeek V3/R1, OLMo 2, Gemma 3, Mistral Small 3.1, Llama 4, Qwen3, SmolLM3, Kimi K2, gpt-oss, GLM-4.5/4.6, and MiniMax-M2.

Key efficiency improvements within this paradigm include:

- **Grouped-query attention (GQA)** -- reduces KV heads to save memory
- **Sliding-window attention** -- limits attention span per layer
- **Multi-head latent attention (MLA)** -- compresses key/value space (used in DeepSeek V3/R1)

These models are proven, well-tooled, and remain the recommended default for building applications, fine-tuning, or experimenting with new algorithms.

---

## Linear Attention Hybrids

### The Quadratic Cost Problem

Traditional scaled-dot-product attention computes an n-by-n attention matrix (where n = sequence length), making it O(n^2) in both time and memory. This becomes a bottleneck for long-context applications.

### Early Linear Attention

Linear attention variants (e.g., *Transformers are RNNs*, 2020) approximated attention using kernel feature maps to avoid computing the full n-by-n matrix, reducing complexity to O(n). However, these early approaches degraded model accuracy and never saw adoption in state-of-the-art LLMs.

### The 2025 Revival

A new wave of hybrid architectures emerged in 2025, mixing linear and full attention layers rather than replacing attention entirely:

| Model | Parameters | Linear Attention Type | Full Attention Type | Ratio (Linear:Full) | Status |
|---|---|---|---|---|---|
| MiniMax-M1 | 456B MoE (46B active) | Lightning Attention | Standard | Most layers | Succeeded by M2 (dropped linear attention) |
| Qwen3-Next | 235B-A22B | Gated DeltaNet | Gated Attention | 3:1 | Active |
| DeepSeek V3.2 | -- | Sparse (subquadratic) | -- | -- | Active |
| Kimi Linear | 48B | Kimi Delta Attention (KDA) | MLA (no RoPE) | 3:1 | Active |

A notable plot twist: MiniMax released M2 *without* linear attention, citing poor accuracy in reasoning and multi-turn tasks. However, Kimi Linear subsequently demonstrated that linear attention can work well with the right design.

### Gated Attention

A simple modification to standard attention: after computing the normal attention output, a **sigmoid gate** modulates the result element-wise. This helps eliminate attention sink and massive activation issues, improving training stability. Used in Qwen3-Next's full-attention layers.

### Gated DeltaNet

The core linear attention mechanism adopted by both Qwen3-Next and Kimi Linear. Key properties:

- **Recurrent state update** -- instead of an n-by-n attention matrix, maintains a fixed-size memory state S updated token-by-token (like an RNN)
- **Alpha (decay gate)** -- controls how fast old memory is forgotten
- **Beta (update gate)** -- controls how strongly new tokens modify the state
- **Output gate** -- controls how much of the output is kept (SiLU activation instead of sigmoid)
- **Delta rule** -- updates memory by computing the difference between new and predicted values

The state update scales linearly with sequence length, but compresses all past context into a fixed-size hidden state, sacrificing some global context modeling. This is why the 3:1 ratio retains some full-attention layers.

### KV Cache Savings

DeltaNet layers do not grow a KV cache. Memory comparison:

```
KV cache (MHA): batch_size x n_tokens x n_heads x d_head x 2 x bytes
KV cache (DeltaNet): batch_size x n_heads x d_head x d_head x bytes
```

The DeltaNet formula has no `n_tokens` dependency -- memory stays constant with context length. The d_head x d_head term is manageable since head dimensions are typically small (e.g., 128 in Qwen3-Next).

### Kimi Linear vs. Qwen3-Next

| Dimension | Qwen3-Next | Kimi Linear |
|---|---|---|
| Linear attention | Gated DeltaNet (scalar gate) | Kimi Delta Attention (channel-wise gate) |
| Full attention | Gated Attention (MHA + sigmoid) | MLA (no RoPE, no sigmoid gate) |
| Positional encoding | RoPE in all layers | NoPE in MLA layers (position handled by KDA blocks) |
| Context length | 262K native | Long-context optimized |
| Efficiency gains | -- | 75% KV cache reduction, up to 6x decoding throughput vs. full attention |

### Outlook

The new generation of linear attention hybrids differs from earlier attempts by combining linear attention with standard attention rather than replacing it entirely. Future work will likely focus on improving long-context stability and reasoning accuracy to close the remaining gap with full-attention models.

---

## Text Diffusion Models

### Core Idea

Instead of generating tokens one at a time (autoregressive), text diffusion models generate all tokens in parallel through iterative denoising -- analogous to how image diffusion models (Stable Diffusion) denoise pixel grids.

- **Corruption**: tokens are progressively masked at random (replacing with [MASK] tokens) rather than adding Gaussian noise to pixels
- **Denoising**: the model learns to predict masked tokens at each step, progressively "unmasking" the sequence
- **Architecture**: typically decoder-style transformers *without* the causal attention mask (bidirectional, like BERT)

The key example is **LLaDA** (Large Language Diffusion Models), an 8B model using the Llama 3 architecture with a generative diffusion objective instead of next-token prediction.

### Autoregressive vs. Diffusion: Trade-offs

| Dimension | Autoregressive LLMs | Diffusion LLMs |
|---|---|---|
| Generation | Sequential (one token at a time) | Parallel (all tokens per denoising step) |
| Steps for 2000 tokens | ~2000 forward passes | ~20-64 denoising steps |
| Token dependencies | Explicit (each token conditioned on all prior) | Implicit (through shared attention across denoising iterations) |
| Streaming | Yes | No |
| Chain-of-thought | Natural | Unclear / difficult |
| Tool calling | Straightforward | Problematic (no sequential chain) |

**The parallel decoding pitfall**: when tokens are sampled independently, the model can produce incoherent outputs. For example, given "Pick a random city: New York, New Orleans, Mexico City, or Panama City?", parallel sampling might independently select "New" and "City" (both high-probability tokens), producing "New City" instead of a valid answer.

### Current State

- **Google's Gemini Diffusion** is reportedly faster than Gemini 2.0 Flash-Lite while maintaining comparable benchmark performance
- Diffusion LLMs are an interesting alternative for smaller, on-device models where speed matters most
- Quality-speed trade-off remains: fewer denoising steps = faster but lower quality; more steps = better quality but approaches autoregressive costs

---

## Code World Models

### From Vision to Code

World models learn an internal simulation of an environment to predict outcomes before acting -- originally developed for vision-based reinforcement learning (Ha & Schmidhuber, 2018). The **Code World Model (CWM)** paper (September 2025) is the first to apply this concept to code, mapping from code to code.

### How CWM Differs from Regular Code LLMs

| Dimension | Regular Code LLMs (e.g., Qwen3-Coder) | Code World Model (CWM) |
|---|---|---|
| Training objective | Next-token prediction on code | Predict resulting program state after code execution |
| Understanding | Static text-level patterns of syntax and logic | Simulates what happens when code runs |
| Output at inference | Code tokens | Code tokens + structured execution traces |
| Architecture | Standard autoregressive transformer | Dense decoder-only transformer (32B params, 131K context, sliding-window attention) |
| Training stages | Pre-training, SFT, RL | Pre-training, **mid-training with world-modeling data**, SFT, RL |

CWM learns to trace how variable states evolve step by step as each line of code executes, effectively building an internal model of program behavior.

### Performance

- On par with gpt-oss-20b (mid reasoning effort) at roughly the same size
- With test-time scaling (best@k with generated unit tests), slightly outperforms gpt-oss-120b (high reasoning effort) while being 4x smaller
- Raschka notes it might be better described as a "world model-augmented LLM" rather than a pure world model

---

## Small Recursive Transformers

### Core Idea

Instead of producing an answer in one forward pass, small recursive transformers **repeatedly refine their output** through iterative self-refinement. Each iteration updates a latent reasoning state, which the authors interpret as the model's "thought" process.

### Hierarchical Reasoning Model (HRM)

- Two small transformer modules (4 blocks each) communicating across recursion levels
- Explicit halting mechanism to decide when to stop iterating
- Achieved a top spot on the ARC challenge
- Operates on grid-based inputs/outputs (not text)

### Tiny Recursive Model (TRM)

TRM is a simpler successor to HRM:

| Dimension | HRM | TRM |
|---|---|---|
| Architecture | Two 4-block transformers | Single 2-layer transformer |
| Parameters | ~28M | ~7M |
| Backpropagation | Only through final few steps | Through all recursive steps |
| Halting | Explicit halting mechanism | Binary cross-entropy loss (learned) |
| Training | Up to 16 refinement steps per batch | Up to 16 refinement steps per batch |
| Training cost | -- | <$500 (4x H100, ~2 days) |

### Surprising Ablation Findings

1. **Fewer layers = better generalization** -- reducing from 4 to 2 layers improved Sudoku accuracy from 79.5% to 87.4%
2. **Attention is not required** -- replacing self-attention with a pure MLP also improved accuracy (74.7% to 87.4%), though only feasible for small, fixed-length contexts

### Limitations

These models are currently specialized for grid-based puzzles (ARC, Sudoku, Maze pathfinding) -- not general text or code. Think of them as efficient "pocket calculators" versus the "computer" that is a general-purpose LLM. However, they could serve as lightweight reasoning modules embedded within larger tool-using LLM systems.

---

## Comparative Summary

| Approach | Goal | Pros | Cons |
|---|---|---|---|
| **Standard Autoregressive Transformers** | General-purpose SOTA | Proven, mature tooling, scaling laws, SOTA performance | Expensive training and inference |
| **Linear Attention Hybrids** | Efficiency at long context | Cuts FLOPs/KV memory for long sequences, compatible with existing transformer tooling | Added complexity, slight accuracy trade-off |
| **Text Diffusion Models** | Parallel generation speed | Better parallelism, fresh approach to text generation | Can't stream, questionable CoT/tool-calling support, not yet SOTA |
| **Code World Models** | Improved code understanding | Verifiable intermediate states, strong coding performance | Execution traces complicate training, code running adds latency |
| **Small Recursive Transformers** | Compact specialized reasoning | Very small (<10M params), strong puzzle generalization, cheap to train | Limited to structured tasks, not general-purpose |

---

## Key Takeaways

1. **Autoregressive transformers remain king.** Standard decoder-style transformers are still the best default choice for building LLM applications -- they are proven, well-tooled, and deliver SOTA performance.

2. **Linear attention hybrids are production-viable.** Kimi Linear demonstrates 75% KV cache reduction and up to 6x decoding throughput by mixing Gated DeltaNet with full attention in a 3:1 ratio, making this the most immediately practical alternative.

3. **The hybrid approach is key.** Unlike earlier linear attention attempts that failed by fully replacing standard attention, the new generation succeeds by combining linear and full attention layers, preserving global context modeling where it matters.

4. **MiniMax's retreat is instructive.** MiniMax dropped linear attention in M2 due to poor reasoning and multi-turn performance, highlighting that linear attention's memory bottleneck remains a real limitation for complex tasks.

5. **Text diffusion trades quality for speed.** Parallel token generation is appealing, but the inability to stream, difficulty with tool calling, and quality degradation from parallel decoding mean diffusion LLMs are best suited for smaller, on-device applications rather than replacing general-purpose LLMs.

6. **Code world models improve understanding, not just pattern matching.** By learning to simulate program execution, CWM achieves performance comparable to models 4x its size (with test-time scaling), suggesting that teaching models *how code runs* is a powerful complement to next-token prediction.

7. **Tiny models can reason impressively.** The 7M-parameter TRM outperforms much larger models on structured reasoning benchmarks, and costs under $500 to train -- a proof that small recursive transformers could serve as specialized reasoning modules within larger systems.

8. **Fewer layers can mean better generalization.** TRM's ablation showing that reducing layers improves accuracy challenges the assumption that deeper is always better, at least for iterative refinement architectures.

---

## References

- Sebastian Raschka's original article: https://magazine.sebastianraschka.com/p/beyond-standard-llms
- The Big LLM Architecture Comparison: https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison
- Attention Is All You Need (2017): https://arxiv.org/abs/1706.03762
- Transformers are RNNs -- Linear Attention (2020): https://arxiv.org/abs/2006.16236
- Gated Delta Networks (2024): https://arxiv.org/abs/2412.06464
- MiniMax-M1 paper: https://arxiv.org/abs/2506.13585
- Kimi Linear paper: https://arxiv.org/abs/2510.26692
- LLaDA -- Large Language Diffusion Models (2025): https://arxiv.org/abs/2502.09992
- Diffusion-LM for Controllable Text Generation (2022): https://arxiv.org/abs/2205.14217
- ParallelBench -- Parallel Decoding Trade-offs: https://arxiv.org/abs/2510.04767
- Gemini Diffusion: https://deepmind.google/models/gemini-diffusion/
- Code World Models (2025): https://www.arxiv.org/abs/2510.02387
- World Models -- Ha & Schmidhuber (2018): https://arxiv.org/abs/1803.10122
- Hierarchical Reasoning Model (HRM): https://arxiv.org/abs/2506.21734
- Tiny Recursive Model (TRM) -- Less is More: https://arxiv.org/abs/2510.04871
- Mixture-of-Recursions (MoR): https://arxiv.org/abs/2507.10524

