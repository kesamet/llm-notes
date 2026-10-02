# Index

Content-oriented catalog of wiki pages.

---

## Sources

| # | Page | Topics | Key terms |
|---|------|--------|-----------|
| 1 | [A Survey on Efficient Inference for Large Language Models](summaries/240414294v3-a-survey-on-efficient-inference-for-large-language-models.md) | Inference optimization taxonomy, quantization, pruning, sparse attention, speculative decoding, offloading, serving systems | KV-cache, operator fusion, dynamic inference, early exit, knowledge distillation |
| 2 | [Intelligent AI Delegation](summaries/260211865v1-intelligent-ai-delegation.md) | AI delegation framework, task decomposition, multi-objective optimization, adaptive coordination, trust/reputation, permission handling | Principal-agent problem, span of control, authority gradient, verifiable task completion |
| 3 | [Dive into Claude Code: Design Space of AI Agent Systems](summaries/260414228v1-dive-into-claude-code-design-space-of-ai-agent-systems.md) | Agent architecture, tool dispatch, permission/safety architecture, MCP extensibility, context construction, memory | Agentic query loop, pre-model context shapers, shell sandboxing, context cost ordering, five-layer decomposition |
| 4 | [From AGI to ASI](summaries/260612683v1-from-agi-to-asi.md) | AGI/ASI definitions, technological pathways, recursive self-improvement, multi-agent coordination, fundamental limits | Universal AI (AIXI), digital intelligence advantages, scaling compute, algorithmic paradigm shifts |
| 5 | [A Technical Tour of the DeepSeek Models from V3 to V3.2](summaries/a-technical-tour-of-the-deepseek-models-from-v3-to-v32.md) | DeepSeek evolution, MLA, MoE, sparse attention, RLVR, GRPO, self-verification | Multi-Head Latent Attention (MLA), Mixture-of-Experts (MoE), DeepSeek Sparse Attention (DSA), Reinforcement Learning with Verifiable Rewards (RLVR), Group Relative Policy Optimization (GRPO), Manifold-Constrained Hyper-Connections (mHC) |
| 6 | [A Visual Guide to Attention Variants in Modern LLMs](summaries/a-visual-guide-to-attention-variants-in-modern-llms.md) | Attention mechanisms comparison, efficiency vs. quality tradeoffs | Multi-Head Attention (MHA), Grouped-Query Attention (GQA), Multi-Head Latent Attention (MLA), Sliding Window Attention (SWA), DeepSeek Sparse Attention (DSA), Gated Attention, Hybrid Attention |
| 7 | [Beyond Standard LLMs](summaries/beyond-standard-llms.md) | Alternative LLM architectures beyond autoregressive transformers | Linear attention hybrids, text diffusion models, code world models, small recursive transformers |
| 8 | [Components of A Coding Agent](summaries/components-of-a-coding-agent.md) | Coding agent harness architecture, repo context, prompt caching, tool use, session memory, subagent delegation | Live repo context, prompt shape, cache reuse, context bloat minimization, bounded subagents, OpenClaw |
| 9 | [From GPT-2 to gpt-oss: Analyzing the Architectural Advances](summaries/from-gpt-2-to-gpt-oss-analyzing-the-architectural-advances.md) | Evolution of transformer architecture, GPT-2 to gpt-oss changes | RoPE, SwiGLU, Mixture-of-Experts (MoE), Grouped Query Attention (GQA), Sliding Window Attention, RMSNorm, MXFP4 quantization, reasoning effort control |
| 10 | [Recent Developments in LLM Architectures: KV Sharing, mHC, and Compressed Attention](summaries/recent-developments-in-llm-architectures-kv-sharing-mhc-and-compressed-attention.md) | Latest architecture advances for long-context inference cost reduction | Cross-Layer KV Sharing, per-layer embeddings, layer-wise attention budgeting, compressed convolutional attention, Manifold-Constrained Hyper-Connections (mHC), CSA, HCA |
| 11 | [The Big LLM Architecture Comparison](summaries/the-big-llm-architecture-comparison.md) | Comprehensive comparison of major open-weight LLM architectures | Attention mechanisms, MoE, normalization strategies, positional encoding, Multi-Token Prediction (MTP) |
| 12 | [Understanding the 4 Main Approaches to LLM Evaluation](summaries/understanding-the-4-main-approaches-to-llm-evaluation-from-scratch.md) | LLM evaluation methodologies, benchmarks, scoring systems | Multiple-choice benchmarks, verification-based evaluation, arena-style leaderboards, LLM-as-a-Judge, Elo rating, Bradley-Terry model, process reward models |
| 13 | [Using Local Coding Agents](summaries/using-local-coding-agents.md) | Local coding agent setup, model selection, harness comparison, Ollama integration | Local LLM, Ollama, Qwen-Code, Codex, Claude Code, speed/memory assessment, SSH tunnel |
| 14 | [Controlling Reasoning Effort in LLMs](summaries/controlling-reasoning-effort-in-llms.md) | Reasoning-effort training recipes across flagship models, training vs. inference scaling | RLVR, GRPO, think tokens, Toggle, on-policy distillation, DeepSeek V4, Nemotron 3 Ultra, Kimi K2.5/K3, GLM-5, Qwen3, Inkling |
| 15 | [GPT-6 Astra: Looped Transformers and Hidden Reasoning](summaries/gpt-6-astra-looped-transformers-and-hidden-reasoning.md) | Looped transformers, hidden reasoning, test-time compute, architectural evolution beyond standard decoder stacks | Looped transformers, hidden reasoning, test-time compute scaling, GPT-6 Astra, internal reasoning loops |

## Concepts

> Concept pages capture abstract ideas, techniques, and mechanisms.
> See `wiki/concepts/` for details.

_To be created: efficient-llm-inference, attention-mechanisms, mixture-of-experts, multi-head-latent-attention, kv-cache-optimization, speculative-decoding, ai-delegation, agent-architecture, coding-agent-harness, local-llm-setup, llm-evaluation, agi-asi_

## Entities

> Entity pages track specific models, organizations, and tools.
> See `wiki/entities/` for details.

_To be created: deepseek, openai, anthropic, meta-ai, qwen, gemma, mistral, llama, claude-code, ollama_

## Comparison tables

> Cross-cutting comparisons organized by theme.
> See `wiki/comparisons/` for details.

_To be created: llm-architecture-comparison, attention-variants-comparison, coding-agent-harness-comparison_

## Syntheses

> Cross-source syntheses integrating insights from multiple pages.
> See `wiki/syntheses/` for details.

_To be created: state-of-open-weight-llms, efficient-inference-landscape, agentic-coding-tools_

## Log

See [log.md](log.md) for chronological activity.

