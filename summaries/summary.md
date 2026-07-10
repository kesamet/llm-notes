# Knowledge Base Summary

> Condensed index of all wiki pages. Refer to individual files for full detail.

---

## 1. A Survey on Efficient Inference for Large Language Models
- **Topics**: Inference optimization taxonomy, quantization, pruning, sparse attention, speculative decoding, offloading, serving systems
- **Key terms**: KV-cache, operator fusion, dynamic inference, early exit, knowledge distillation
- **Models covered**: General survey across LLMs

## 2. Intelligent AI Delegation
- **Topics**: AI delegation framework, task decomposition, multi-objective optimization, adaptive coordination, trust/reputation, permission handling
- **Key terms**: Principal-agent problem, span of control, authority gradient, verifiable task completion

## 3. Dive into Claude Code: Design Space of AI Agent Systems
- **Topics**: Agent architecture, tool dispatch, permission/safety architecture, MCP extensibility, context construction, memory
- **Key terms**: Agentic query loop, pre-model context shapers, shell sandboxing, context cost ordering, five-layer decomposition

## 4. From AGI to ASI
- **Topics**: AGI/ASI definitions, technological pathways, recursive self-improvement, multi-agent coordination, fundamental limits
- **Key terms**: Universal AI (AIXI), digital intelligence advantages, scaling compute, algorithmic paradigm shifts

## 5. A Technical Tour of the DeepSeek Models from V3 to V3.2
- **Topics**: DeepSeek evolution, MLA, MoE, sparse attention, RLVR, GRPO, self-verification
- **Models**: DeepSeek V3, V3.1, V3.2-Exp, V3.2, DeepSeekMath V2
- **Key terms**: Multi-Head Latent Attention (MLA), Mixture-of-Experts (MoE), DeepSeek Sparse Attention (DSA), Reinforcement Learning with Verifiable Rewards (RLVR), Group Relative Policy Optimization (GRPO), Manifold-Constrained Hyper-Connections (mHC)

## 6. A Visual Guide to Attention Variants in Modern LLMs
- **Topics**: Attention mechanisms comparison, efficiency vs. quality tradeoffs
- **Key terms**: Multi-Head Attention (MHA), Grouped-Query Attention (GQA), Multi-Head Latent Attention (MLA), Sliding Window Attention (SWA), DeepSeek Sparse Attention (DSA), Gated Attention, Hybrid Attention

## 7. Beyond Standard LLMs
- **Topics**: Alternative LLM architectures beyond autoregressive transformers
- **Key terms**: Linear attention hybrids, text diffusion models, code world models, small recursive transformers
- **Models**: DeepSeek R1, Llama 4, Qwen3, OLMo 2, Gemma 3, Mistral Small 3.1, SmolLM3, Kimi K2, gpt-oss, GLM-4.5/4.6, MiniMax-M2

## 8. Components of A Coding Agent
- **Topics**: Coding agent harness architecture, repo context, prompt caching, tool use, session memory, subagent delegation
- **Key terms**: Live repo context, prompt shape, cache reuse, context bloat minimization, bounded subagents, OpenClaw

## 9. From GPT-2 to gpt-oss: Analyzing the Architectural Advances
- **Topics**: Evolution of transformer architecture, GPT-2 to gpt-oss changes
- **Key terms**: RoPE, SwiGLU, Mixture-of-Experts (MoE), Grouped Query Attention (GQA), Sliding Window Attention, RMSNorm, MXFP4 quantization, reasoning effort control
- **Models**: GPT-2, gpt-oss, Qwen3

## 10. Recent Developments in LLM Architectures: KV Sharing, mHC, and Compressed Attention
- **Topics**: Latest architecture advances for long-context inference cost reduction
- **Key terms**: Cross-Layer KV Sharing, per-layer embeddings, layer-wise attention budgeting, compressed convolutional attention, Manifold-Constrained Hyper-Connections (mHC), CSA, HCA
- **Models**: Gemma 4 E2B/E4B, Laguna XS.2, ZAYA1-8B, DeepSeek V4

## 11. The Big LLM Architecture Comparison
- **Topics**: Comprehensive comparison of major open-weight LLM architectures
- **Key terms**: Attention mechanisms, MoE, normalization strategies, positional encoding, Multi-Token Prediction (MTP)
- **Models**: DeepSeek V3/R1, OLMo 2, Gemma 3, Mistral Small 3.1, Llama 4, Qwen3, SmolLM3, Kimi K2, GPT-OSS, Grok 2.5, GLM-4.5, Qwen3-Next, MiniMax-M2

## 12. Understanding the 4 Main Approaches to LLM Evaluation
- **Topics**: LLM evaluation methodologies, benchmarks, scoring systems
- **Key terms**: Multiple-choice benchmarks, verification-based evaluation, arena-style leaderboards, LLM-as-a-Judge, Elo rating, Bradley-Terry model, process reward models

## 13. Using Local Coding Agents
- **Topics**: Local coding agent setup, model selection, harness comparison, Ollama integration
- **Key terms**: Local LLM, Ollama, Qwen-Code, Codex, Claude Code, speed/memory assessment, SSH tunnel
- **Models**: Qwen3.6 35B-A3B, North Mini Code 1.0

