# Index

Content-oriented catalog of wiki pages.

---

## Sources

| # | Page | Topics | Key terms |
|---|------|--------|-----------|
| 1 | [A Survey on Efficient Inference for Large Language Models](wiki/01-efficient-inference-for-llms.md) | Inference optimization taxonomy, quantization, pruning, sparse attention, speculative decoding, offloading, serving systems | KV-cache, operator fusion, dynamic inference, early exit, knowledge distillation |
| 2 | [Intelligent AI Delegation](wiki/02-intelligent-ai-delegation.md) | AI delegation framework, task decomposition, multi-objective optimization, adaptive coordination, trust/reputation, permission handling | Principal-agent problem, span of control, authority gradient, verifiable task completion |
| 3 | [Dive into Claude Code: Design Space of AI Agent Systems](wiki/03-claude-code-design-space.md) | Agent architecture, tool dispatch, permission/safety architecture, MCP extensibility, context construction, memory | Agentic query loop, pre-model context shapers, shell sandboxing, context cost ordering, five-layer decomposition |
| 4 | [From AGI to ASI](wiki/04-from-agi-to-asi.md) | AGI/ASI definitions, technological pathways, recursive self-improvement, multi-agent coordination, fundamental limits | Universal AI (AIXI), digital intelligence advantages, scaling compute, algorithmic paradigm shifts |
| 5 | [A Technical Tour of the DeepSeek Models from V3 to V3.2](wiki/05-deepseek-v3-to-v3-2.md) | DeepSeek evolution, MLA, MoE, sparse attention, RLVR, GRPO, self-verification | Multi-Head Latent Attention (MLA), Mixture-of-Experts (MoE), DeepSeek Sparse Attention (DSA), Reinforcement Learning with Verifiable Rewards (RLVR), Group Relative Policy Optimization (GRPO), Manifold-Constrained Hyper-Connections (mHC) |
| 6 | [A Visual Guide to Attention Variants in Modern LLMs](wiki/06-attention-variants.md) | Attention mechanisms comparison, efficiency vs. quality tradeoffs | Multi-Head Attention (MHA), Grouped-Query Attention (GQA), Multi-Head Latent Attention (MLA), Sliding Window Attention (SWA), DeepSeek Sparse Attention (DSA), Gated Attention, Hybrid Attention |
| 7 | [Beyond Standard LLMs](wiki/07-beyond-standard-llms.md) | Alternative LLM architectures beyond autoregressive transformers | Linear attention hybrids, text diffusion models, code world models, small recursive transformers |
| 8 | [Components of A Coding Agent](wiki/08-components-of-a-coding-agent.md) | Coding agent harness architecture, repo context, prompt caching, tool use, session memory, subagent delegation | Live repo context, prompt shape, cache reuse, context bloat minimization, bounded subagents, OpenClaw |
| 9 | [From GPT-2 to gpt-oss: Analyzing the Architectural Advances](wiki/09-gpt2-to-gpt-oss.md) | Evolution of transformer architecture, GPT-2 to gpt-oss changes | RoPE, SwiGLU, Mixture-of-Experts (MoE), Grouped Query Attention (GQA), Sliding Window Attention, RMSNorm, MXFP4 quantization, reasoning effort control |
| 10 | [Recent Developments in LLM Architectures: KV Sharing, mHC, and Compressed Attention](wiki/10-recent-llm-architectures.md) | Latest architecture advances for long-context inference cost reduction | Cross-Layer KV Sharing, per-layer embeddings, layer-wise attention budgeting, compressed convolutional attention, Manifold-Constrained Hyper-Connections (mHC), CSA, HCA |
| 11 | [The Big LLM Architecture Comparison](wiki/11-big-llm-architecture-comparison.md) | Comprehensive comparison of major open-weight LLM architectures | Attention mechanisms, MoE, normalization strategies, positional encoding, Multi-Token Prediction (MTP) |
| 12 | [Understanding the 4 Main Approaches to LLM Evaluation](wiki/12-llm-evaluation-approaches.md) | LLM evaluation methodologies, benchmarks, scoring systems | Multiple-choice benchmarks, verification-based evaluation, arena-style leaderboards, LLM-as-a-Judge, Elo rating, Bradley-Terry model, process reward models |
| 13 | [Using Local Coding Agents](wiki/13-local-coding-agents.md) | Local coding agent setup, model selection, harness comparison, Ollama integration | Local LLM, Ollama, Qwen-Code, Codex, Claude Code, speed/memory assessment, SSH tunnel |

## Concepts

- [Efficient LLM Inference](concepts/efficient-llm-inference.md)
- [Attention Mechanisms](concepts/attention-mechanisms.md)
- [Mixture-of-Experts (MoE)](concepts/mixture-of-experts.md)
- [Multi-Head Latent Attention (MLA)](concepts/multi-head-latent-attention.md)
- [KV Cache Optimization](concepts/kv-cache-optimization.md)
- [Speculative Decoding](concepts/speculative-decoding.md)
- [AI Delegation](concepts/ai-delegation.md)
- [Agent Architecture](concepts/agent-architecture.md)
- [Coding Agent Harness](concepts/coding-agent-harness.md)
- [Local LLM Setup](concepts/local-llm-setup.md)
- [LLM Evaluation](concepts/llm-evaluation.md)
- [AGI / ASI](concepts/agi-asi.md)

## Entities

- [DeepSeek](entities/deepseek.md)
- [OpenAI](entities/openai.md)
- [Anthropic](entities/anthropic.md)
- [Meta AI](entities/meta-ai.md)
- [Qwen](entities/qwen.md)
- [Gemma](entities/gemma.md)
- [Mistral](entities/mistral.md)
- [Llama](entities/llama.md)
- [Claude Code](entities/claude-code.md)
- [Ollama](entities/ollama.md)

## Comparisons tables

- [LLM architecture comparison](comparisons/llm-architecture-comparison.md)
- [Attention variants comparison](comparisons/attention-variants-comparison.md)
- [Coding agent harness comparison](comparisons/coding-agent-harness-comparison.md)

## Syntheses

- [State of open-weight LLMs](syntheses/state-of-open-weight-llms.md)
- [Efficient inference landscape](syntheses/efficient-inference-landscape.md)
- [Agentic coding tools](syntheses/agentic-coding-tools.md)

## Log

See [log.md](log.md) for chronological activity.

