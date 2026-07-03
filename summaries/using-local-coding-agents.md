# Using Local Coding Agents -- Wiki

> Based on Sebastian Raschka's article (Jun 2026)
> Source: https://magazine.sebastianraschka.com/p/using-local-coding-agents

---

## Table of Contents

- [Overview](#overview)
- [Why Local?](#why-local)
- [Recommended Models](#recommended-models)
- [Qwen3.6 35B-A3B](#qwen36-35b-a3b)
- [North Mini Code 1.0](#north-mini-code-10)
- [Harness Overview](#harness-overview)
- [Local LLM Setup with Ollama](#local-llm-setup-with-ollama)
- [Speed and Memory Assessment](#speed-and-memory-assessment)
- [Benchmark Performance Assessment](#benchmark-performance-assessment)
- [Agent Codebase Audit](#agent-codebase-audit)
- [Qwen-Code Setup](#qwen-code-setup)
- [Codex Setup](#codex-setup)
- [Claude Code Setup](#claude-code-setup)
- [Harness Comparison](#harness-comparison)
- [Remote Model via SSH Tunnel](#remote-model-via-ssh-tunnel)
- [Key Takeaways](#key-takeaways)
- [References](#references)

---

## Overview

This article is a practical tutorial on assembling a fully local coding agent: a locally served open-weight LLM connected to a coding harness that reads files, applies edits, runs commands, and verifies changes. For architecture background on what a coding harness does, see the companion wiki page on [Components of A Coding Agent](./components-of-a-coding-agent.md), which covers the six core components of harness design in depth.

The article covers three harnesses (Qwen-Code, Codex, Claude Code) connected to locally served models via Ollama, including security auditing considerations, speed/quality assessment methodology, and token usage comparisons across all three harnesses.

---

## Why Local?

| Motivation | Detail |
|---|---|
| **Cost predictability** | Fixed hardware cost; immune to API price changes and subscription throttling |
| **Privacy** | Prompts and files stay on-machine; important for sensitive data (e.g., receipts, proprietary code) |
| **Reproducibility** | Model stays fixed; cloud model upgrades can silently break workflows |
| **Offline use** | No internet dependency; works on flights, cabins, air-gapped environments |
| **Transparency** | Full access to harness code; can audit and modify behavior |

The author notes that Anthropic was observed throttling flagship model performance for LLM research, and proprietary services may become more restrictive over time.

---

## Recommended Models

### Qwen3.6 35B-A3B

- **Architecture**: Mixture-of-Experts with hybrid attention (similar to Qwen3-Coder and Qwen3.5; see [Beyond Standard LLMs](./beyond-standard-llms.md) for hybrid attention details)
- **Size**: ~22 GB download; 30--40 GB RAM at runtime
- **Speed**: ~40 tok/sec on Mac Mini M4; ~30 tok/sec on DGX Spark
- **Ollama tag**: `qwen3.6:35b-mlx` (Mac/Apple Silicon) or `qwen3.6:35b` (Linux)
- **Strengths**: Best-in-class on most coding benchmarks at this size; specifically optimized for Qwen-Code harness

The `*-mlx` variant uses Apple's Metal Performance Shaders and is strongly preferred on Apple Silicon Macs.

### North Mini Code 1.0

- **Size**: Similar to Qwen3.6 35B-A3B
- **Speed**: Slightly ahead of Qwen3.6 in speed when using the same quantization level (Q4)
- **Strengths**: Strongest alternative at this size class; slightly better in speed benchmarks but marginally behind Qwen3.6 on most coding tasks

### Model Comparison

| Model | RAM (50k ctx) | Gen Speed (Mac) | Gen Speed (DGX) | Coding Benchmark |
|---|---|---|---|---|
| `qwen3.6:35b-mlx` | ~29 GB | ~40 tok/sec | -- | Best-in-class |
| `qwen3.6:35b` (Q4) | ~29 GB | -- | ~30 tok/sec | Best-in-class |
| North Mini Code 1.0 (Q4) | ~29 GB | Slightly faster | Similar | Very close second |
| `gemma4:e2b` | ~8 GB | Faster | Faster | Insufficient for agent use |

Anything above 20--30 tok/sec is considered reasonable for local agent work -- roughly on par with GPT 5.5 at "high" reasoning effort.

---

## Harness Overview

The article focuses on three harnesses and one model-specific harness:

| Harness | Open Source | Model-Specific Optimization | Notes |
|---|---|---|---|
| **Qwen-Code** | Yes | Qwen models (Qwen-Code harness) | Recommended pairing for Qwen models |
| **Codex CLI** | Yes | OpenAI GPT (but supports Ollama) | Uses least tokens; good universal harness |
| **Claude Code** | No | Anthropic Claude | Proprietary codebase; uses most tokens |

For completeness, the article also mentions OpenCode, Cline, Pi, Noumena Code, OpenClaw, and Hermes -- but OpenClaw/Hermes are general agent platforms suited for multi-tool workflows rather than terminal coding specifically. For that distinction, see [Components of A Coding Agent](./components-of-a-coding-agent.md#comparison-with-openclaw).

The key claim from Nvidia's *Polar: Agentic RL on Any Harness at Scale* (May 2026) is that the Qwen3.5-4B base model performs best in the Qwen-Code harness. However, the author's own small benchmark (Section 8) found that Qwen3.6 actually performs *better* in Codex than in Qwen-Code, which complicates the "use the model's native harness" assumption.

---

## Local LLM Setup with Ollama

**Ollama** is used as the serving engine because it has minimal setup, cross-platform support (macOS, Linux, Windows), and supports the OpenAI API standard (required for connecting to all three harnesses).

### Installing and Pulling a Model

```bash
# macOS (Apple Silicon) -- prefer MLX variant
ollama pull qwen3.6:35b-mlx

# Linux
ollama pull qwen3.6:35b
```

Test the model is running via the Ollama GUI or:

```bash
ollama run qwen3.6:35b-mlx
# Exit with /bye
```

Ollama optionally supports cloud-hosted open-weight models (e.g., GLM 5.2) at similar pricing to ChatGPT/Claude, useful for models too large for consumer hardware.

---

## Speed and Memory Assessment

Run the `ollama_speed_memory_bench.py` script from https://github.com/rasbt/local-coding-agent-evals:

```bash
# macOS
uv run speed-memory-benchmark/ollama_speed_memory_bench.py --model qwen3.6:35b-mlx

# Linux
uv run speed-memory-benchmark/ollama_speed_memory_bench.py --model qwen3.6:35b
```

The script sends prompts ranging from 1k to 50k words, measures prefill speed, generation speed, and RAM consumption across context lengths. Key findings:

- For 50k context windows, Qwen3.6 and North Mini Code use up to 30 GB RAM
- Generation speed stays stable across context lengths (~40 tok/sec Mac, ~30 tok/sec DGX)
- RSS RAM reports are inaccurate on macOS for MLX models -- use Activity Monitor instead
- Minimum viable agent speed is roughly 20--30 tok/sec; both recommended models clear this bar easily

---

## Benchmark Performance Assessment

Before committing to a model for agent work, check standard benchmarks (technical report, model hub, https://artificialanalysis.ai/models/) and run a tool-calling benchmark.

The `ollama_hard_reasoning_bench.py` script tests tool-calling judgment -- not just code generation:

```
qwen3.6:35b → 3/5 (60%) -- passes conceptual debugging and security review; struggles with "what file/action first" agentic judgment
north-mini-code → 1/5 (20%) -- multiple tool-choice errors and JSON formatting failures
gemma4:e2b → 0/5 (0%) -- wrong tool selection throughout; not suitable for autonomous agent use
```

The author recommends also curating a **personal task set** reflecting your common coding workflows, including difficult cases encountered during real projects, as a living benchmark for evaluating future models.

---

## Agent Codebase Audit

Before running any coding harness that can read files and execute commands, audit the open-source codebase. The author recommends asking a trusted agent (e.g., Claude Code or Codex with a frontier model) with a focused prompt covering:

- Install scripts and package lifecycle hooks
- Shell command execution boundaries
- File read/write scope at runtime
- Secret handling and environment variable inheritance
- Influence surfaces: repo instructions, tool output, MCP tools, extensions
- Network calls, telemetry, and data egress
- Update mechanisms post-installation

Key findings for Qwen-Code specifically:

| Risk Area | Finding | vs. Standard Practice |
|---|---|---|
| Shell execution | Runs commands with strict approval gates; `--yolo` disables them | Expected for coding agents |
| Data egress | Sends usage telemetry and metadata to Alibaba/Aliyun endpoints by default | Similar to Codex and Claude Code |
| File/secret boundaries | Workspace files readable; writes require approval with overwrite protections | Standard |
| Prompt injection | Repo instructions, tool output, MCP, and project config can steer the agent | Normal; treat untrusted repos as hostile |

### Disabling Telemetry (Qwen-Code)

Create `~/.qwen/settings.json`:

```json
{
"privacy": { "usageStatisticsEnabled": false },
"telemetry": { "enabled": false, "logPrompts": false },
"outboundCorrelation": { "propagateTraceContext": false },
"general": { "enableAutoUpdate": false },
"tools": { "approvalMode": "default", "sandbox": true },
"mcpServers": {},
"hooks": { "disableAllHooks": true }
}
```

Setting `enableAutoUpdate: false` trades automatic security patches for explicit control over when new code is pulled. Cline, Codex, and Claude Code have similar telemetry defaults that also require explicit disabling.

The author also notes that Claude Code's codebase is proprietary -- it cannot be audited -- and it appears to send data to both Anthropic and Datadog.

---

## Qwen-Code Setup

### Installation

Option 1 (fast, trusts published artifact):
```bash
npm install -g @qwen-code/qwen-code@latest
```

Option 2 (build from source, for auditability):
```bash
git clone https://github.com/QwenLM/qwen-code.git
cd qwen-code
npm install
npm run build
mkdir -p ~/.local/bin
cat > ~/.local/bin/qwen <<'SH'
#!/usr/bin/env sh
exec "$HOME/Developer/qwen-code/scripts/cli-entry.js" "$@"
SH
chmod +x ~/.local/bin/qwen
export PATH="$HOME/.local/bin:$PATH"
qwen --version
```

### Connecting to Local Ollama

1. Run `qwen` and choose "Custom Provider"
2. Choose "OpenAI-compatible" (Ollama uses the OpenAI API standard)
3. Set base URL: `http://127.0.0.1:11434/v1`
4. Enter `ollama` as the API key placeholder
5. Select downloaded models (check `ollama list`)
6. Enable thinking mode for better reasoning

### Adding New Models

After `ollama pull <model>`, edit `~/qwen/settings.json` and copy an existing model entry, changing `"id"` and `"name"` to the new Ollama model name. Switch models in-session with `/model`.

### Updating (source build route)

```bash
cd ~/Developer/qwen-code
git pull && npm install && npm run build
qwen --version
```

---

## Codex Setup

Codex CLI supports local Ollama models. Create a separate config `~/.codex/ollama.config.toml`:

```toml
model = "qwen3.6:35b"
model_provider = "ollama"
model_reasoning_effort = "high"
personality = "pragmatic"

[projects."/home/rasbt"]
trust_level = "trusted"
```

Run with the regular GPT profile:
```bash
codex
```

Run with the local Ollama profile:
```bash
codex --profile ollama
```

This lets both modes coexist without any manual model switching.

---

## Claude Code Setup

Claude Code does not have a dedicated local-provider configuration path like Codex. The recommended integration uses Ollama's native Claude integration:

```bash
ollama launch claude --model qwen3.6:35b
```

For use with a remote Ollama server (e.g., DGX via SSH tunnel):
```bash
OLLAMA_HOST=http://127.0.0.1:11434 ollama launch claude --model qwen3.6:35b
```

---

## Harness Comparison

The article ran all three harnesses against the same 5-task agent capability benchmark using Qwen3.6, North Mini Code, Nemotron 3 Nano, and Gemma 4 E2B.

### Task Success Rate

| Model | Qwen-Code | Codex | Claude Code |
|---|---|---|---|
| Qwen3.6 35B-A3B | 4/5 | **5/5** | **5/5** |
| North Mini Code 1.0 | 4/5 | -- | **5/5** |
| Nemotron 3 Nano | 4/5 | -- | **5/5** |
| Gemma 4 E2B | 1/5 | -- | 2/5 |

Qwen3.6 surprisingly performs *better* in Codex and Claude Code than in its native Qwen-Code harness, suggesting that model-harness optimization claims (e.g., "Qwen3.6 is optimized for Qwen-Code") may not hold in practice for capable models.

### Token Usage

| Harness | Average Input Tokens | Average Output Tokens | Relative Speed |
|---|---|---|---|
| **Codex** | Lowest | Low | Fastest |
| **Qwen-Code** | Medium | Medium | Medium |
| **Claude Code** | Highest (~578k input / 25 turns) | ~4.5k | Slowest |

Claude Code's high token usage is driven by **input tokens**, not output. The harness repeatedly feeds back prior messages, tool calls, command outputs, and file contents across turns, accumulating a large prompt-side history. This matches the architecture described in [Components of A Coding Agent -- Component 4](./components-of-a-coding-agent.md#component-4-minimizing-context-bloat) as the cost of not aggressively compressing the transcript.

The implication: if two harnesses achieve equal task success, the harness using 50% fewer tokens will run tasks roughly 2x faster (for a locally served model where throughput is the bottleneck).

---

## Remote Model via SSH Tunnel

To run the coding harness on a Mac while the Ollama model is hosted on a separate machine (e.g., DGX Spark):

1. Quit the local Ollama app on Mac (or change port)
2. Verify Ollama is not running locally: `curl http://127.0.0.1:11434/v1/models` (should return empty)
3. Open an SSH tunnel from the Mac:

```bash
ssh -N -L 11434:127.0.0.1:11434 <username>@<remote-host>
```

4. Verify the tunnel works: `curl http://127.0.0.1:11434/v1/models` (should return DGX models)
5. All three harnesses (Qwen-Code, Codex, Claude Code) will now use the remote model transparently

Keep the terminal running the `ssh -N -L ...` command open while using any harness. Press `Ctrl-C` to stop the tunnel.

---

## Key Takeaways

1. **The harness shapes the experience more than the model.** All three capable models (Qwen3.6, North Mini Code, Nemotron 3 Nano) reached 5/5 in Claude Code -- differences in task success are largely a function of harness quality and context management, not model intelligence alone. See [Components of A Coding Agent](./components-of-a-coding-agent.md) for what specifically makes harnesses different.

2. **Qwen3.6 and North Mini Code are production-viable at ~40 tok/sec.** This is roughly the same perceived speed as GPT 5.5 at "high" reasoning, making local models a practical daily-driver alternative rather than just a curiosity.

3. **Model-native harness optimization is not guaranteed.** Despite Qwen3.6 being advertised as optimized for Qwen-Code, it scored higher in Codex and Claude Code on the author's benchmark. If you already have muscle memory with one harness, it may be reasonable to just try plugging the local model into it.

4. **Audit before running.** Coding agents have a large blast radius -- they read files, run shell commands, and can exfiltrate data through approved tools even when using a local LLM. Review data-egress defaults and disable telemetry before first use.

5. **Fewer tokens = faster tasks (for local models).** Token usage is primarily driven by the harness, not the model. Codex uses the fewest tokens; Claude Code uses the most. For locally served models where throughput is limited, this directly translates to wall-clock speed.

6. **Use Codex `--profile` for clean multi-mode workflows.** Setting up a separate `ollama.config.toml` lets you run GPT-backed Codex and local-model Codex side by side without any manual switching -- a clean approach for comparing models or falling back to a frontier model when needed.

7. **SSH tunneling makes remote GPUs seamless.** Running Ollama on a DGX or workstation while using the harness on a Mac is straightforward via SSH port forwarding, and requires no changes to harness configuration.

---

## References

- Article source: https://magazine.sebastianraschka.com/p/using-local-coding-agents
- Local coding agent eval suite: https://github.com/rasbt/local-coding-agent-evals
- Qwen-Code harness: https://github.com/QwenLM/qwen-code
- Codex CLI: https://github.com/openai/codex
- Ollama: https://ollama.com/
- Ollama download page: https://ollama.com/download/
- Ollama Claude Code integration: https://docs.ollama.com/integrations/claude-code
- Polar: Agentic RL on Any Harness at Scale (May 2026): https://arxiv.org/abs/2605.24220
- North Mini Code report (Cohere, Jun 2026): https://huggingface.co/blog/CohereLabs/introducing-north-mini-code
- Artificial Analysis model benchmarks: https://artificialanalysis.ai/models/
- Pi harness: https://github.com/earendil-works/pi
- Build a Large Language Model (From Scratch): https://amzn.to/4fqvn0D
- Build a Reasoning Model (From Scratch): https://mng.bz/Nwr7

