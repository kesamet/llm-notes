# Dive into Claude Code: The Design Space of Today's and Future AI Agent Systems -- Wiki

> Based on Jiacheng Liu, Xiaohan Zhao, Xinyi Shang, and Zhiqiang Shen's paper (April 2026)
> Source: arXiv:2604.14228v1

---

- [Overview](#overview)
- [Five Design Values](#five-design-values)
- [Thirteen Design Principles](#thirteen-design-principles)
- [Architecture Overview](#architecture-overview)
- [Seven Components](#seven-components)
- [Five-Layer Decomposition](#five-layer-decomposition)
- [Four Core Design Questions](#four-core-design-questions)
- [The Agentic Query Loop](#the-agentic-query-loop)
- [Turn Execution Pipeline](#turn-execution-pipeline)
- [Tool Dispatch and Streaming](#tool-dispatch-and-streaming)
- [Pre-Model Context Shapers](#pre-model-context-shapers)
- [Recovery Mechanisms](#recovery-mechanisms)
- [Permission and Safety Architecture](#permission-and-safety-architecture)
- [Seven Permission Modes](#seven-permission-modes)
- [Authorization Pipeline](#authorization-pipeline)
- [Auto-Mode Classifier](#auto-mode-classifier)
- [Shell Sandboxing](#shell-sandboxing)
- [Extensibility: MCP, Plugins, Skills, and Hooks](#extensibility-mcp-plugins-skills-and-hooks)
- [Four Extension Mechanisms](#four-extension-mechanisms)
- [Tool Pool Assembly](#tool-pool-assembly)
- [Context Cost Ordering](#context-cost-ordering)
- [Context Construction and Memory](#context-construction-and-memory)
- [Context Window Assembly](#context-window-assembly)
- [CLAUDE.md Hierarchy](#claudemd-hierarchy)
- [Five-Layer Compaction Pipeline](#five-layer-compaction-pipeline)
- [Subagent Delegation and Orchestration](#subagent-delegation-and-orchestration)
- [Built-in Subagent Types](#built-in-subagent-types)
- [Isolation Architecture](#isolation-architecture)
- [Sidechain Transcripts](#sidechain-transcripts)
- [Session Persistence and Recovery](#session-persistence-and-recovery)
- [Comparative Analysis: Claude Code vs. OpenClaw](#comparative-analysis-claude-code-vs-openclaw)
- [Architectural Trade-offs and Tensions](#architectural-trade-offs-and-tensions)
- [Open Directions for Future Agent Systems](#open-directions-for-future-agent-systems)
- [Key Takeaways](#key-takeaways)
- [References](#references)

---

## Overview

This paper is a source-level architectural analysis of Claude Code (v2.1.88), Anthropic's agentic coding tool, based on the publicly available TypeScript source (~1,884 files, ~512K lines). The study identifies five human values that motivate the architecture, traces them through thirteen design principles, and documents the implementation across seven components and five subsystem layers. It also compares Claude Code's architecture with OpenClaw, an open-source multi-channel AI assistant gateway, to show how the same recurring design questions produce different architectural answers under different deployment contexts.

For a higher-level overview of what coding agents are and how their harnesses work, see the existing wiki page on "Components of a Coding Agent." This page goes deeper into the specific architectural decisions, source-level implementation details, and design trade-offs documented in the paper.

---

## Five Design Values

The architecture is motivated by five values identified from Anthropic's documentation and design statements:

| Value | Core Concern | Key Evidence |
|---|---|---|
| **Human Decision Authority** | Human retains ultimate control via a principal hierarchy (Anthropic > operators > users) | 93% permission approval rate led to restructuring safety around defined boundaries rather than per-action approvals |
| **Safety, Security, and Privacy** | Protect humans/code/data even when human is inattentive | Auto-mode threat model targets four risk categories: overeager behavior, honest mistakes, prompt injection, model misalignment |
| **Reliable Execution** | Do what the human meant; stay coherent over time | Three-phase loop: gather context, take action, verify results |
| **Capability Amplification** | Increase what humans can accomplish per unit of effort | ~27% of Claude Code-assisted tasks were work that would not have been attempted without the tool |
| **Contextual Adaptability** | Fit the user's specific context; relationship improves over time | Auto-approve rates increase from ~20% at <50 sessions to >40% by 750 sessions |

The paper also applies a sixth concern -- **long-term human capability preservation** -- as an evaluative lens. It notes that AI-assisted developers scored 17% lower on comprehension tests (Shen and Tamkin, 2026), and a causal analysis found code complexity increased by 40.7% after Cursor adoption (He et al., 2025).

---

## Thirteen Design Principles

| Principle | Values Served | Design Question Answered |
|---|---|---|
| Deny-first with human escalation | Authority, Safety | Should unrecognized actions be allowed, blocked, or escalated? |
| Graduated trust spectrum | Authority, Adaptability | Fixed permission level, or a spectrum users traverse over time? |
| Defense in depth with layered mechanisms | Safety, Authority, Reliability | Single safety boundary, or multiple overlapping ones? |
| Externalized programmable policy | Safety, Authority, Adaptability | Hardcoded policy, or externalized configs with lifecycle hooks? |
| Context as scarce resource with progressive management | Reliability, Capability | Binding constraint managed via single-pass truncation or graduated pipeline? |
| Append-only durable state | Reliability, Authority | Mutable state, checkpoint snapshots, or append-only logs? |
| Minimal scaffolding, maximal operational harness | Capability, Reliability | Invest in scaffolding-side reasoning, or operational infrastructure? |
| Values over rules | Capability, Authority | Rigid decision procedures, or contextual judgment with deterministic guardrails? |
| Composable multi-mechanism extensibility | Capability, Adaptability | One unified extension API, or layered mechanisms at different context costs? |
| Reversibility-weighted risk assessment | Capability, Safety | Same oversight for all actions, or lighter for reversible/read-only ones? |
| Transparent file-based configuration and memory | Adaptability, Authority | Opaque database, embedding-based retrieval, or user-visible version-controllable files? |
| Isolated subagent boundaries | Reliability, Safety, Capability | Subagents share parent's context and permissions, or operate in isolation? |
| Graceful recovery and resilience | Reliability, Capability | Fail hard on errors, or silently recover and reserve human attention for unrecoverable situations? |

---

## Architecture Overview

### Seven Components

1. **User**: Submits prompts, approves permissions, reviews output
2. **Interfaces**: Interactive CLI, headless CLI (`claude -p`), Agent SDK, IDE/Desktop/Browser -- all feed the same loop
3. **Agent loop**: `queryLoop()` async generator in `query.ts`
4. **Permission system**: Deny-first rule evaluation, auto-mode ML classifier, hook-based interception
5. **Tools**: Up to 54 built-in tools (19 unconditional + 35 conditional) via `assembleToolPool()`, merged with MCP tools
6. **State & persistence**: Append-only JSONL session transcripts, global prompt history, subagent sidechain files
7. **Execution environment**: Shell (with optional sandboxing), filesystem, web fetching, MCP servers, remote execution

### Five-Layer Decomposition

| Layer | Responsibility | Key Source Locations |
|---|---|---|
| **Surface** | Entry points and rendering | `src/entrypoints/`, `src/screens/`, `src/components/` (ink framework) |
| **Core** | Agent loop, compaction pipeline | `query.ts`, `query/` directory |
| **Safety/Action** | Permissions, hooks, extensibility, tools, sandbox, subagents | `permissions.ts`, `types/hooks.ts`, `tools.ts`, `AgentTool.tsx` |
| **State** | Context assembly, runtime state, persistence, memory, sidechains | `context.ts`, `src/state/`, `sessionStorage.ts`, `claudemd.ts` |
| **Backend** | Execution backends, external resources | `BashTool.tsx`, `src/remote/`, `services/mcp/client.ts`, 42 tool subdirectories |

### Four Core Design Questions

| Question | Claude Code's Answer | Alternatives |
|---|---|---|
| Where does reasoning live? | Model reasons; harness executes. Only ~1.6% of codebase is AI decision logic, ~98.4% is operational infrastructure | Devin: explicit planning structures; LangGraph: developer-defined state graphs |
| How many execution engines? | Single `queryLoop()` regardless of interface (CLI, SDK, IDE) | Mode-specific engines for surface-specific optimization |
| Default safety posture? | Deny-first with human escalation; multiple independent safety layers in parallel | SWE-Agent/OpenHands: Docker isolation; Aider: Git-based rollback |
| Binding resource constraint? | Context window (200K-1M tokens); five compaction strategies execute before every model call | Compute budget (limiting model calls) or explicit scratchpad |

---

## The Agentic Query Loop

### Turn Execution Pipeline

Each turn follows a fixed sequence in `query.ts`:

1. **Settings resolution** -- destructure immutable parameters (system prompt, user context, permission callback, model config)
2. **Mutable state initialization** -- single State object; seven continue sites overwrite via whole-object assignment
3. **Context assembly** -- `getMessagesAfterCompactBoundary()` retrieves messages from last compact boundary forward
4. **Pre-model context shapers** -- five shapers execute sequentially
5. **Model call** -- `for await` loop over `deps.callModel()` streams response
6. **Tool-use dispatch** -- if response contains `tool_use` blocks, flow to tool orchestration
7. **Permission gate** -- each tool request passes through the permission system
8. **Tool execution and result collection** -- results added as `tool_result` messages; loop continues
9. **Stop condition** -- if response contains no `tool_use` blocks (text only), turn is complete

The loop follows the **ReAct pattern** (Yao et al., 2022): model generates reasoning + tool invocations, harness executes actions, results feed the next iteration.

### Tool Dispatch and Streaming

Two execution paths:

- **Primary**: `StreamingToolExecutor` -- begins executing tools as they stream in, reducing latency for multi-tool responses
- **Fallback**: `runTools()` in `toolOrchestration.ts` -- iterates over partitions

Coordination mechanisms:
- **Sibling abort controller**: fires when any Bash tool errors, terminating other in-flight subprocesses
- **Progress-available signal**: wakes the consumer when new output is ready
- Results buffered and emitted in received order (model expects tool results in same order as requests)

Classification: read-only operations run in parallel; state-modifying operations are serialized.

### Pre-Model Context Shapers

Five shapers execute sequentially before every model call, each progressively more aggressive:

| # | Shaper | Mechanism | Gate |
|---|---|---|---|
| 1 | **Budget reduction** | Per-message size limits on tool results; replaces oversized outputs with content references | Always active |
| 2 | **Snip** | Lightweight trim removing older history segments | `HISTORY_SNIP` flag |
| 3 | **Microcompact** | Fine-grained compression; time-based + optional cache-aware path | `CACHED_MICROCOMPACT` flag |
| 4 | **Context collapse** | Read-time projection over conversation history (virtual view, no mutation) | `CONTEXT_COLLAPSE` flag |
| 5 | **Auto-compact** | Full model-generated summary via `compactConversation()` | Enabled by default (configurable) |

### Recovery Mechanisms

- **Max output tokens escalation**: retry with escalated limit (up to 3 attempts per turn)
- **Reactive compaction**: when context is near capacity, summarize just enough to free space (fires at most once per turn)
- **Prompt-too-long handling**: attempts context-collapse overflow recovery and reactive compaction before terminating
- **Streaming fallback**: handles streaming API issues
- **Fallback model**: enables switching to alternative model if primary fails

---

## Permission and Safety Architecture

Seven independent safety layers; a request must pass through all applicable layers:

1. **Tool pre-filtering** -- blanket-denied tools removed from model's view before any call
2. **Deny-first rule evaluation** -- deny rules always take precedence over allow rules (even when allow is more specific)
3. **Permission mode constraints** -- active mode determines baseline handling for unmatched requests
4. **Auto-mode classifier** -- ML-based; potentially denies requests the rule system would allow
5. **Shell sandboxing** -- approved shell commands may still execute inside a sandbox
6. **Not restoring permissions on resume** -- session-scoped permissions are not restored on resume/fork
7. **Hook-based interception** -- `PreToolUse` hooks can modify permission decisions

### Seven Permission Modes

| Mode | Behavior | Type |
|---|---|---|
| `plan` | Model creates plan; execution only after user approval | External |
| `default` | Standard interactive; most operations require approval | External |
| `acceptEdits` | Edits auto-approved; other shell commands require approval | External |
| `auto` | ML classifier evaluates requests not passing fast-path checks | Conditional (feature-gated) |
| `dontAsk` | No prompting, but deny rules still enforced | External |
| `bypassPermissions` | Skips most prompts; safety-critical checks still apply | External |
| `bubble` | Internal-only for subagent permission escalation to parent terminal | Internal |

Modes span a **graduated autonomy spectrum** from `plan` (most restrictive) to `bypassPermissions` (most permissive).

### Authorization Pipeline

1. **Pre-filtering**: `filterToolsByDenyRules()` strips blanket-denied tools at assembly time
2. **PreToolUse hook**: can return `permissionDecision` (deny/ask) or `updatedInput`; allow does not bypass subsequent checks
3. **Rule evaluation**: deny-first engine; MCP tools matched by fully qualified `mcp__server__tool` name
4. **Permission handler** branches into four paths:
- **Coordinator**: for multi-agent coordination; automated resolution before user interaction
- **Swarm worker**: worker agents with own resolution logic
- **Speculative classifier**: races pre-started classification against timeout for instant approval
- **Interactive**: standard user approval dialog (fallback)

When the classifier or deny rule blocks an action, the model receives the denial reason, revises its approach, and tries a safer alternative in the next loop iteration.

### Auto-Mode Classifier

Implemented in `yoloClassifier.ts`. When `TRANSCRIPT_CLASSIFIER` is enabled, loads three prompt resources (base system prompt, external permissions template, internal template for Anthropic users). Evaluates proposed tool invocation against conversation transcript, producing allow/deny/manual-approval.

### Shell Sandboxing

`shouldUseSandbox()` provides filesystem and network isolation independent of the application-level permission model. Authorization (permission system) and isolation (sandbox) operate on different axes -- a command can be permission-approved but still sandboxed.

---

## Extensibility: MCP, Plugins, Skills, and Hooks

### Four Extension Mechanisms

| Mechanism | Unique Capability | Context Cost | Insertion Point |
|---|---|---|---|
| **MCP servers** | External service integration (multi-transport: stdio, SSE, HTTP, WebSocket, SDK, IDE-specific) | High (tool schemas) | `model()`: tool pool |
| **Plugins** | Multi-component packaging + distribution (10 component types) | Medium (varies) | All three injection points |
| **Skills** | Domain-specific instructions + meta-tool invocation (SKILL.md with 15+ frontmatter fields) | Low (descriptions only) | `assemble()`: context injection |
| **Hooks** | Lifecycle interception + event-driven automation (27 event types, 4 command types) | Zero by default | `execute()`: pre/post tool |

The three injection points in the agent loop:
- **assemble()**: controls what the model sees (CLAUDE.md, skill descriptions, MCP resources, hook context)
- **model()**: controls what it can reach (built-in tools, MCP tools, SkillTool, AgentTool)
- **execute()**: controls whether/how an action runs (permission rules, PreToolUse/PostToolUse hooks, Stop hook)

### Tool Pool Assembly

`assembleToolPool()` in `tools.ts` follows a five-step pipeline:

1. **Base tool enumeration** -- `getAllBaseTools()` returns up to 54 tools (19 always + 35 conditional)
2. **Mode filtering** -- `getTools()` applies mode-specific filtering (e.g., `CLAUDE_CODE_SIMPLE` mode: only Bash, Read, Edit)
3. **Deny rule pre-filtering** -- `filterToolsByDenyRules()` strips blanket-denied tools
4. **MCP tool integration** -- MCP tools filtered by deny rules and merged with built-in tools
5. **Deduplication** -- by name, built-in tools taking precedence over MCP tools

### Context Cost Ordering

The graduated context-cost ordering (zero for hooks, low for skills, medium for plugins, high for MCP) means cheap extensions scale widely without exhausting the context window. This is why four mechanisms exist rather than a single unified extension API.

---

## Context Construction and Memory

### Context Window Assembly

Sources assembled (in loading order):

1. System prompt (+ output style modifications)
2. Environment info via `getSystemContext()` -- git status (memoized once per session)
3. CLAUDE.md hierarchy via `getUserContext()` -- four-level instruction files (memoized)
4. Path-scoped rules -- load lazily when agent reads files in matching directories
5. Auto memory -- contextually relevant entries prefetched asynchronously
6. Tool metadata -- skill descriptions, MCP tool names, deferred tool definitions (via ToolSearch)
7. Conversation history -- subject to compaction
8. Tool results -- file reads, command outputs, subagent summaries
9. Compact summaries -- replace older history segments

CLAUDE.md content is delivered as **user context** (a user message), not system prompt content. This means model compliance with these instructions is **probabilistic** rather than guaranteed -- the permission system provides the deterministic enforcement layer.

### CLAUDE.md Hierarchy

Four levels:

1. **Managed memory** (e.g., `/etc/claude-code/CLAUDE.md`): OS-level policy for all users
2. **User memory** (`~/.claude/CLAUDE.md`): private global instructions
3. **Project memory** (`CLAUDE.md`, `.claude/CLAUDE.md`, `.claude/rules/*.md`): checked into codebase
4. **Local memory** (`CLAUDE.local.md`): gitignored, private project-specific

Loading order: later-loaded files receive more model attention (reverse priority). Files below CWD load lazily. Supports `@include` directive for modular instruction sets.

Memory retrieval uses an **LLM-based scan of memory-file headers** to select up to five relevant files on demand -- no embeddings or vector similarity index.

### Five-Layer Compaction Pipeline

The five shapers implement a **lazy-degradation principle**: apply least disruptive compression first, escalating only when cheaper strategies prove insufficient.

The `buildPostCompactMessages()` output structure: `[boundaryMarker, ...summaryMessages, ...messagesToKeep, ...attachments, ...hookResults]`. Compaction never modifies or deletes previously written transcript lines -- it only appends new boundary and summary events (mostly-append design).

---

## Subagent Delegation and Orchestration

### Built-in Subagent Types

Up to six built-in types (depending on feature flags):

| Type | Role |
|---|---|
| **Explore** | Read/search-oriented investigation; write/edit tools in deny-list |
| **Plan** | Creates structured plans; execution through standard permission model |
| **General-purpose** | Broadly capable; used when explicitly requested |
| **Claude Code Guide** | Onboarding and documentation assistance |
| **Verification** | Runs validation checks (test suites, linting) |
| **Statusline-setup** | Terminal status line configuration |

Custom subagents via `.claude/agents/*.md` files or plugin contributions. Markdown body serves as system prompt; YAML frontmatter specifies config (tools, model, effort, permissionMode, mcpServers, hooks, maxTurns, etc.).

**Key distinction**: `SkillTool` injects instructions into the current context window; `AgentTool` spawns a new, isolated one.

### Isolation Architecture

| Mode | Mechanism |
|---|---|
| **Worktree** | Temporary git worktree; own repository copy without affecting parent |
| **Remote** (internal-only) | Remote Claude Code environment; always runs in background |
| **In-process** (default) | Shares filesystem with parent; isolated conversation context |

Permission override logic: when a subagent defines a `permissionMode`, the override is applied unless the parent is already in `bypassPermissions`, `acceptEdits`, or `auto` mode (user decisions take precedence).

Two-tier permission scoping: SDK-level permissions (`--allowedTools`) preserved across all agents; session-level rules replaced with subagent's declared `allowedTools` when explicitly provided.

### Sidechain Transcripts

Each subagent writes its own `.jsonl` + `.meta.json` transcript file. Only the subagent's final response text and metadata return to the parent context -- full subagent history never enters parent's context window. Agent teams consume approximately **7x the tokens** of a standard session in plan mode, making summary-only return critical.

Multi-instance coordination uses **file locking** (no message broker or distributed coordination): zero external dependencies, full debuggability via plain-text JSON files.

---

## Session Persistence and Recovery

Three independent persistence channels:

1. **Session transcripts**: append-only JSONL files (project-scoped, one per session) storing messages, compaction markers, file-history snapshots, attribution snapshots
2. **Global prompt history**: user prompts only in `history.jsonl` (supports Up-arrow and ctrl+r navigation)
3. **Subagent sidechains**: separate `.jsonl` + `.meta.json` per subagent

**Resume** (`--resume`): rebuilds conversation by replaying transcript. **Fork**: creates new session from existing one. Neither restores session-scoped permissions (deliberate safety-conservative choice -- trust is always established in the current session).

File-history checkpoints for `--rewind-files` stored at `~/.claude/file-history/<sessionId>/` (file-level snapshots for reverting filesystem changes).

---

## Comparative Analysis: Claude Code vs. OpenClaw

| Dimension | Claude Code | OpenClaw |
|---|---|---|
| **System scope** | CLI/IDE coding harness; ephemeral per-session process | Persistent WebSocket gateway daemon; multi-channel control plane |
| **Trust model** | Deny-first per-action rule evaluation; 7 permission modes; ML classifier | Single trusted operator per gateway; DM pairing + allowlists; opt-in sandboxing (Docker/SSH/OpenShell) |
| **Agent runtime** | `queryLoop()` async generator as system center | Pi-agent runner embedded inside gateway RPC dispatch; per-session queue serialization |
| **Extension architecture** | 4 mechanisms at graduated context costs | Manifest-first plugin system with 12 capability types + central registry; separate skills layer; MCP via `openclaw mcp` |
| **Memory and context** | CLAUDE.md 4-level hierarchy; 5-layer compaction; LLM-based memory scan | Workspace bootstrap files (AGENTS.md, SOUL.md, etc.); hybrid vector + keyword search; experimental "dreaming" for long-term memory promotion |
| **Multi-agent** | Task-delegating subagents (Explore, Plan, general-purpose); worktree isolation; summary-only return | Multi-agent routing with isolated agents + sub-agent delegation with configurable nesting depth (max 5, default 1); thread-bound sessions |

Key observations:
- The same design questions recur across different agent systems; answers vary with deployment context
- The two make opposite bets: per-action safety evaluation vs. perimeter-level access control; agent loop as center vs. gateway as center
- They are **composable**: OpenClaw can host Claude Code via ACP (Agent Client Protocol)

---

## Architectural Trade-offs and Tensions

| Value Pair | Tension | Evidence |
|---|---|---|
| Authority x Safety | Approval fatigue vs. protection | 93% approval rate undermines human vigilance; safety must compensate via classifier and sandboxing |
| Safety x Capability | Performance vs. defense depth | >50-subcommand fallback skips per-subcommand deny checks due to parsing overhead |
| Adaptability x Safety | Extensibility vs. attack surface | CVEs exploit pre-trust initialization of hooks and MCP servers |
| Capability x Adaptability | Proactivity vs. disruption | 12-18% more tasks but preference drops at high frequencies |
| Capability x Reliability | Velocity vs. coherence | Bounded context prevents full codebase awareness; subagent isolation limits cross-agent consistency |

Notable empirical findings:
- AI tools made experienced developers **19% slower** despite a perceived 20% improvement (Becker et al., 2025)
- Code complexity increased by **40.7%** after Cursor adoption (He et al., 2025)
- ~25% of AI-introduced issues persisted to latest revision; security issues persisted at higher rate (Liu et al., 2026)

---

## Open Directions for Future Agent Systems

1. **Silent failure and observability-evaluation gap**: 78% of AI failures are invisible (Bessemer 2026); 89% observability adoption vs. 52.4% offline evaluation. Generator-evaluator separation and post-hoc checks likely needed at the harness layer.

2. **Cross-session persistence and longitudinal relationships**: gap between static instructions (CLAUDE.md) and single-session transcripts. Need for accumulating experiential memory that survives restarts. OpenClaw's "dreaming" system is an early attempt.

3. **Harness boundary evolution**: Four axes of expansion:
- **Where**: virtualizing session/harness/sandbox into independently replaceable interfaces (Managed Agents)
- **When**: proactive agents (KAIROS with tick-based heartbeats; +12-18% task completion but preference penalty at high frequency)
- **What**: vision-language-action models extending beyond textual tool returns
- **With whom**: role-differentiated multi-agent systems and multi-agent debate

4. **Horizon scaling**: extending reliable execution from turns/sessions to multi-session programs (days/weeks). Autonomous research pipelines (Lu et al., 2024; Gottweis et al., 2025; Novikov et al., 2025).

5. **Governance and oversight at scale**: EU AI Act (fully applicable August 2026); only 13.3% of agentic systems publish agent-specific safety cards. Need for externally auditable logging/transparency interfaces.

6. **Long-term human capability preservation**: treating the sustainability gap as a first-class design problem rather than a downstream evaluation metric. No production agent currently provides per-session signals for comprehension or convention drift.

---

## Key Takeaways

1. Claude Code's architecture is ~98.4% deterministic operational infrastructure and ~1.6% AI decision logic. The design philosophy is "minimal scaffolding, maximal operational harness" -- the harness creates conditions for the model to decide well, rather than constraining its choices.

2. The core agent loop is a simple while-true cycle (`queryLoop()` in `query.ts`) following the ReAct pattern. Most engineering complexity lives in the surrounding subsystems: safety (7 layers), context management (5 compaction layers), extensibility (4 mechanisms), and delegation (subagents with isolation).

3. The permission system implements a **deny-first** posture with a **graduated trust spectrum** -- motivated by the finding that 93% of permission prompts are approved (rendering interactive confirmation behaviorally unreliable as a sole safety mechanism).

4. Context is treated as the binding resource constraint. The five-layer compaction pipeline (budget reduction, snip, microcompact, context collapse, auto-compact) implements lazy degradation -- applying the least disruptive compression first before escalating.

5. The four extension mechanisms (MCP, plugins, skills, hooks) are differentiated by **context cost**: zero for hooks, low for skills, medium for plugins, high for MCP servers. This answers why a single unified extension API is insufficient.

6. Subagent delegation uses **summary-only return** to the parent context -- preserving the context-as-bottleneck principle at the cost of requiring self-contained prompts (no inherited conversation history by default).

7. The architecture reveals a structural tension: good local decisions by the model can produce poor global outcomes when bounded context prevents global awareness. Empirical evidence from adjacent tools (40.7% complexity increase, 19% slowdown) is consistent with this architectural prediction.

8. The most consequential open question identified by the paper is not how to add more autonomy, but how to preserve long-term human capability -- a concern the current architecture does not address as a first-class design problem.

---

## References

- Paper: https://arxiv.org/abs/2604.14228
- GitHub: https://github.com/VILA-Lab/Dive-into-Claude-Code
- Claude Code docs: https://code.claude.com/docs
- Anthropic safe agents framework: https://www.anthropic.com/news/our-framework-for-developing-safe-and-trustworthy-agents
- Auto-mode analysis (Hughes, 2026): https://www.anthropic.com/engineering/claude-code-auto-mode
- Sandboxing (Dworken and Weller-Davies, 2025): https://www.anthropic.com/engineering/claude-code-sandboxing
- Building effective agents (Schluntz and Zhang, 2024): https://www.anthropic.com/research/building-effective-agents
- Managed Agents (Martin et al., 2026): https://www.anthropic.com/engineering/managed-agents
- Harness design (Rajasekaran, 2026): https://anthropic.com/engineering/harness-design-long-running-apps
- Agent autonomy measurement (McCain et al., 2026): https://anthropic.com/research/measuring-agent-autonomy
- AI impact at Anthropic (Huang et al., 2025): https://anthropic.com/research/how-ai-is-transforming-work-at-anthropic
- OpenClaw: https://github.com/openclaw/openclaw
- MCP security survey (Hou et al., 2025): ACM TOSEM
- Cursor complexity study (He et al., 2025): arXiv:2511.04427
- AI-generated code debt (Liu et al., 2026): arXiv:2603.28592
- Developer productivity RCT (Becker et al., 2025): arXiv:2507.09089
- LangChain state of agent engineering (2026): https://www.langchain.com/state-of-agent-engineering

