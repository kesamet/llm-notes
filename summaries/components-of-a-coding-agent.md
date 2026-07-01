# Components of A Coding Agent -- Wiki

> Based on Sebastian Raschka's article (Apr 2026)
> Source: https://magazine.sebastianraschka.com/p/components-of-a-coding-agent

---

## Table of Contents

- [Overview](#overview)
- [LLMs, Reasoning Models, and Agents](#llms-reasoning-models-and-agents)
- [The Coding Harness](#the-coding-harness)
- [Component 1: Live Repo Context](#component-1-live-repo-context)
- [Component 2: Prompt Shape and Cache Reuse](#component-2-prompt-shape-and-cache-reuse)
- [Component 3: Tool Access and Use](#component-3-tool-access-and-use)
- [Component 4: Minimizing Context Bloat](#component-4-minimizing-context-bloat)
- [Component 5: Structured Session Memory](#component-5-structured-session-memory)
- [Component 6: Delegation With Bounded Subagents](#component-6-delegation-with-bounded-subagents)
- [Comparison With OpenClaw](#comparison-with-openclaw)
- [Key Takeaways](#key-takeaways)
- [References](#references)

---

## Overview

Coding agents like Claude Code and Codex CLI are not just LLMs -- they are LLMs wrapped in an **agentic coding harness** that manages repo context, tool use, prompt caching, memory, and long-session continuity. Much of their apparent capability comes from this harness, not the model alone. The article identifies **six core components** that make a coding harness effective, illustrated through the author's [Mini Coding Agent](https://github.com/rasbt/mini-coding-agent).

The key thesis: a good coding harness can make even a non-reasoning model feel much stronger than it does in a plain chat box, because the surrounding system -- not just the model -- shapes most of the user experience.

---

## LLMs, Reasoning Models, and Agents

These three concepts are often conflated but are distinct layers:

| Concept | Definition | Analogy |
|---|---|---|
| **LLM** | The raw next-token model | The engine |
| **Reasoning model** | An LLM trained/prompted to spend more inference-time compute on intermediate reasoning and verification | A beefed-up engine |
| **Agent** | A control loop around the model that decides what to inspect, which tools to call, how to update state, and when to stop | The driver using the engine |

Related terms:

| Term | Scope |
|---|---|
| **Agent harness** | The software scaffold around an agent -- manages context, tools, prompts, state, and control flow (general-purpose) |
| **Coding harness** | A task-specific agent harness for software engineering -- manages code context, tools, execution, and iterative feedback |

A better LLM provides a better foundation for a reasoning model, and a harness gets more out of this reasoning model. But coding work is only partly about next-token generation -- much of it is repo navigation, search, function lookup, diff application, test execution, error inspection, and keeping relevant information in context.

---

## The Coding Harness

The harness is the software layer around the model that:

- Assembles prompts
- Exposes tools
- Tracks file state
- Applies edits
- Runs commands
- Manages permissions
- Caches stable prefixes
- Stores memory

Since modern frontier LLMs (e.g., GPT-5.4, Opus 4.6, GLM-5) have broadly similar vanilla capabilities, the **harness is often the distinguishing factor** that makes one LLM product feel more capable than another. The author speculates that dropping a top open-weight model like GLM-5 into a comparable harness could match the proprietary products.

---

## Component 1: Live Repo Context

**Problem**: When a user says "fix the tests," the model needs to know the repo structure, current branch, project conventions, and what instructions exist (e.g., AGENTS.md, README).

**Solution**: The harness collects a **workspace summary** of "stable facts" upfront before doing any work:

| Information Collected | Why It Matters |
|---|---|
| Git repo root and layout | Directs file search to the right locations |
| Current branch and status | Reveals what changes are in progress |
| Project documents (AGENTS.md, README) | Contains instructions, test commands, conventions |
| Recent commits | Provides context on current work focus |

**Takeaway**: The agent starts with a pre-built workspace summary rather than starting from zero on every prompt. Without this context, "fix the tests" is not a self-contained instruction.

---

## Component 2: Prompt Shape and Cache Reuse

**Problem**: Coding sessions are repetitive -- agent rules, tool descriptions, and workspace summaries stay mostly the same across turns. Rebuilding the entire prompt from scratch each turn wastes compute.

**Solution**: Split the prompt into a **stable prefix** and **dynamic suffix**:

| Layer | Contents | Update Frequency |
|---|---|---|
| **Stable prompt prefix** | General instructions, tool descriptions, workspace summary | Rarely (reused across turns) |
| **Dynamic suffix** | Short-term memory, recent transcript, latest user request | Every turn |

The stable prefix is **cached** so the runtime doesn't reprocess it on every interaction. Only the dynamic portion changes per turn.

**Takeaway**: Efficient prompt construction is not about gathering information (that's Component 1) -- it's about packaging and caching that information for repeated model calls with minimal redundant compute.

---

## Component 3: Tool Access and Use

**Problem**: A plain model can suggest commands in prose, but a coding agent needs to actually **execute** commands and retrieve results.

**Solution**: The harness provides a pre-defined list of named tools with structured inputs and validation:

### Tool-Use Flow

1. **Model emits** a structured action (tool name + arguments)
2. **Harness validates**: Is this a known tool? Are arguments valid? Is the path inside the workspace?
3. **Approval gate**: Does this action need user confirmation?
4. **Execution**: Only runs after all checks pass
5. **Result bounded**: Output is clipped/formatted before feeding back into the loop

### Typical Tool Set

| Tool | Purpose |
|---|---|
| List files | Directory navigation |
| Read file | Inspect source code |
| Search/grep | Find symbols and patterns |
| Write/edit file | Modify source code |
| Run shell command | Execute tests, builds, arbitrary commands |

### Safety Boundaries

- File access is restricted to the workspace (path checking)
- Malformed actions are rejected before execution
- Dangerous operations require explicit user approval
- The model operates within a constrained action space, which paradoxically **improves reliability** by preventing totally arbitrary commands

**Takeaway**: The harness gives the model less freedom but more usability. Structured tool use with validation is what separates an agent from a chatbot that pastes shell suggestions.

---

## Component 4: Minimizing Context Bloat

**Problem**: Coding agents accumulate massive context through repeated file reads, lengthy tool outputs, and logs. Long contexts are expensive and introduce noise from irrelevant information.

**Solution**: Two main compaction strategies:

### Strategy 1: Clipping

Shortens individual pieces of text that would otherwise consume the prompt budget:

- Long document snippets
- Large tool outputs
- Verbose memory notes
- Transcript entries

### Strategy 2: Transcript Reduction

Compresses the full session history into a smaller promptable summary using these techniques:

| Technique | How It Works |
|---|---|
| **Recency weighting** | Keep recent events richer (more likely relevant); compress older events aggressively |
| **Deduplication** | Remove repeated file reads -- the model doesn't need to see the same file content multiple times |
| **Summarization** | Convert detailed history into compact summaries |

**Takeaway**: This is one of the most underrated parts of coding-agent design. A lot of apparent "model quality" is really **context quality** -- what the model sees determines what it can do.

---

## Component 5: Structured Session Memory

**Problem**: The agent needs both a complete record of everything that happened (for resumption) and a compact, distilled view of what currently matters (for task continuity).

**Solution**: Separate state into two layers:

| Layer | Purpose | Contents | Behavior |
|---|---|---|---|
| **Full transcript** | Durable record; enables session resumption | All user requests, tool outputs, LLM responses | Append-only; stored as JSON on disk |
| **Working memory** | Task continuity; distilled current state | Current task, important files, recent notes | Modified and compacted (not just appended) |

### Distinction From Compact Transcript (Component 4)

| | Compact Transcript | Working Memory |
|---|---|---|
| **Job** | Prompt reconstruction -- give the model a compressed view of recent history | Task continuity -- maintain explicitly what matters across turns |
| **Content** | Compressed recent events | Curated task state (current task, key files, notes) |
| **Emphasis** | Compression, clipping, recency | Explicit maintenance and modification |

Both the full transcript and working memory are typically stored as **JSON files on disk**, enabling session resumption if the agent is closed and restarted.

---

## Component 6: Delegation With Bounded Subagents

**Problem**: The main agent may need side answers (which file defines a symbol, what a config says, why a test is failing) while in the middle of a primary task. Forcing one loop to carry every thread of work creates bottleneck and context overload.

**Solution**: Delegate bounded subtasks to **subagents** that inherit enough context to be useful but run inside tighter boundaries than the main agent.

### Subagent Design Constraints

| Dimension | Main Agent | Subagent |
|---|---|---|
| **Tool access** | Full (read + write + execute) | Often restricted (e.g., read-only) |
| **Recursion** | Can spawn subagents | Typically cannot spawn further subagents |
| **Context** | Full workspace + transcript | Inherited subset relevant to the subtask |
| **Execution** | Synchronous loop | Can run in parallel (implementation-dependent) |

### Implementation Differences

- **Claude Code**: Has supported subagents for a long time; subagents can be constrained by tool access and recursion depth
- **Codex**: Added subagents more recently; does not generally force read-only mode; boundaries focus on task scoping, context, and depth rather than strict permission restrictions

**Takeaway**: The tricky design problem is not just how to spawn a subagent but how to **bind** one -- ensuring it has enough context to be useful without duplicating work, touching the same files, or spawning unbounded recursion.

---

## Comparison With OpenClaw

OpenClaw is not a direct equivalent to coding agents like Claude Code or Codex -- it's a **general agent platform** that can also code, rather than a specialized terminal coding assistant.

| Dimension | Coding Agents (Claude Code, Codex) | OpenClaw |
|---|---|---|
| **Optimization target** | Person working in a repository | Running many long-lived local agents |
| **Primary use case** | Inspect files, edit code, run local tools | Agents across chats, channels, and workspaces |
| **Coding** | Core focus | One workload among several |

Despite different emphases, OpenClaw shares several harness features:

- Workspace instruction files (AGENTS.md, SOUL.md, TOOLS.md)
- JSONL session files with transcript compaction
- Session management and subagent spawning

---

## Key Takeaways

1. **The harness matters as much as the model.** A good coding harness makes a model feel much stronger than it does in a plain chat box. Since frontier LLMs have broadly similar vanilla capabilities, the harness is often the distinguishing factor.

2. **Live repo context is foundational.** The agent must understand the workspace (git state, project docs, directory structure) before it can meaningfully act on instructions like "fix the tests."

3. **Prompt caching is an efficiency multiplier.** Separating stable information (instructions, tools, workspace summary) from dynamic information (user request, recent transcript) avoids redundant reprocessing across turns.

4. **Structured tool use > arbitrary command generation.** Constraining the model to validated, named tools with approval gates improves both safety and reliability.

5. **Context quality drives perceived model quality.** Clipping, deduplication, and transcript compression are unglamorous but critical -- what the model sees determines what it can do.

6. **Two-layer memory separates recording from reasoning.** A full transcript for durability and a working memory for task continuity solve different problems and should be maintained independently.

7. **Subagent design is about bounding, not just spawning.** The hard problem is constraining subagents to be useful without creating runaway recursion, file conflicts, or duplicated work.

---

## References

- Mini Coding Agent source code: https://github.com/rasbt/mini-coding-agent
- Mini Coding Agent main file: https://github.com/rasbt/mini-coding-agent/blob/main/mini_coding_agent.py
- Visual Guide to Attention Variants: https://magazine.sebastianraschka.com/p/visual-attention-variants
- Build a Large Language Model (From Scratch): https://amzn.to/4fqvn0D
- Build a Reasoning Model (From Scratch): https://mng.bz/Nwr7

