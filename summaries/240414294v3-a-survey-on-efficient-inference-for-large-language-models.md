# A Survey on Efficient Inference for Large Language Models -- Wiki

> Based on Zhou et al.'s survey paper (July 2024)
> Source: arXiv:2404.14294v3

---

## Table of Contents

- [Overview and Taxonomy](#overview-and-taxonomy)
- [Efficiency Bottleneck Analysis](#efficiency-bottleneck-analysis)
- [Data-Level Optimization](#data-level-optimization)
- [Input Compression](#input-compression)
- [Output Organization](#output-organization)
- [Model-Level Optimization](#model-level-optimization)
- [Quantization](#quantization)
- [Weight Pruning and Sparsification](#weight-pruning-and-sparsification)
- [Sparse Attention Patterns](#sparse-attention-patterns)
- [Low-Rank Factorization and NAS](#low-rank-factorization-and-nas)
- [Knowledge Distillation](#knowledge-distillation)
- [Dynamic Inference and Early Exit](#dynamic-inference-and-early-exit)
- [Efficient Architecture Design](#efficient-architecture-design)
- [System-Level Optimization](#system-level-optimization)
- [Operator and Graph Optimization](#operator-and-graph-optimization)
- [Speculative Decoding](#speculative-decoding)
- [Offloading Strategies](#offloading-strategies)
- [Serving Systems](#serving-systems)
- [Experimental Comparisons](#experimental-comparisons)
- [Application Scenarios and Future Directions](#application-scenarios-and-future-directions)
- [Key Takeaways](#key-takeaways)
- [References](#references)

---

## Overview and Taxonomy

This survey organizes the landscape of efficient LLM inference into a three-level hierarchy. Each level targets a different stage of the inference pipeline, from how data enters the model, to how the model itself is structured and compressed, to how the system executes computation.

| Level | Category | Sub-categories |
|-------|----------|----------------|
| Data-level | Input compression | Prompt pruning, prompt summary, soft prompt, RAG |
| Data-level | Output organization | Parallel decoding, DAG structures, DSLs |
| Model-level | Efficient structure | MoE, efficient attention, Transformer alternates |
| Model-level | Model compression | Quantization, pruning, NAS/LRF, distillation, early exit |
| System-level | Inference engine | Operator fusion, speculative decoding, offloading |
| System-level | Serving systems | Memory management, batching, scheduling, distribution |

The survey's key contribution is unifying these techniques under a common analytical framework that traces inefficiency back to three root causes (model size, attention complexity, auto-regressive decoding) and maps each technique to the specific bottleneck it addresses.

---

## Efficiency Bottleneck Analysis

The paper identifies **three root causes** of inefficient LLM inference, each dominating a different regime of operation.

**1. Model Size (Memory Capacity Bottleneck)**

`LLaMA-70B` requires 140 GB VRAM in FP16, meaning deployment needs either 6x RTX 3090Ti (24 GB each) or 2x A100 (80 GB each). Model weights must be loaded from HBM for every generated token during decoding.

**2. Attention Complexity (Compute Bottleneck)**

Attention has O(n^2 * d) complexity in input length n during the **prefilling** stage. For long-context applications (32K-128K tokens), this becomes the dominant cost.

**3. Auto-regressive Decoding (Memory Bandwidth Bottleneck)**

Each generated token requires loading all model weights from HBM. The **KV cache** grows linearly with sequence length, consuming additional memory. Single-token generation cannot saturate GPU compute units.

**Two-Stage Inference Model**

| Stage | Bound | Characteristic | Optimization Target |
|-------|-------|----------------|---------------------|
| Prefilling | Compute-bound | Processes all input tokens in parallel | Reduce FLOPs (efficient attention, W+A quantization) |
| Decoding | Memory-bound | Generates one token per step via KV cache | Reduce memory access (weight-only quant, speculative decoding) |

This two-stage distinction is critical: a technique that accelerates prefilling may not help decoding, and vice versa. Weight-only quantization primarily benefits decoding (fewer bytes transferred per step), while weight-activation quantization targets prefilling (lower-precision compute).

---

## Data-Level Optimization

### Input Compression

Input compression reduces the number of tokens processed during prefilling, directly attacking the O(n^2) attention cost.

**Prompt Pruning** removes unimportant tokens or sentences before they reach the model. Methods differ in granularity and selection criteria:

| Method | Granularity | Selection Criterion |
|--------|-------------|---------------------|
| Selective Context | sentence/token | self-information score |
| LLMLingua | token | perplexity from small LM |
| LongLLMLingua | token | question-aware contrastive perplexity |
| STDC | sentence | parse tree distance |
| PCRL | token | RL-trained policy |
| CoT-Influx | token | learnable pruner for chain-of-thought |

**Prompt Summary** uses a trained model to compress the input into a shorter natural-language equivalent. **RECOMP** trains both extractive and abstractive compressors supervised by whether the compressed prompt preserves answer quality. **SemanticCompression** segments by topic then summarizes each segment.

**Soft Prompt Compression** maps long inputs into a small number of learned continuous embedding vectors, bypassing the discrete token bottleneck entirely. **Gisting** fine-tunes the LLM to compress instructions into "gist tokens" at the boundary between system and user prompts. **ICAE** uses a LoRA-adapted encoder to compress context into memory slots, achieving 4x token reduction with minimal quality loss. **AutoCompressors** train the LLM itself to produce summary vectors iteratively.

**RAG as Compression** retrieves only relevant passages rather than stuffing entire documents into context. See methods like FLARE (forward-looking active retrieval), REPLUG (ensemble of retrieved docs), and Self-RAG (critic tokens for retrieval decisions).

### Output Organization

Rather than compressing input, these methods restructure the generation process to enable parallelism.

**Skeleton-of-Thought (SoT)** prompts the LLM to first produce a point-by-point skeleton, then expands each point independently and in parallel. Achieves up to **2.39x speedup** on answer-organization tasks (e.g., recommendations, explanations). Not suitable for highly sequential reasoning.

**SGD** extends SoT by representing sub-problems as a directed acyclic graph, enabling partial parallelism even when some points depend on others.

**APAR** fine-tunes LLMs to emit `[fork]` and `[join]` control tokens during generation, allowing the serving system to spawn parallel decode branches dynamically. Achieves **1.4-2.0x speedup** without requiring the skeleton-first approach.

**SGLang** provides a domain-specific language for composing LLM calls with automatic dependency analysis, KV cache reuse across calls, and constrained decoding. Particularly effective for multi-call pipelines (agents, tree-of-thought).

---

## Model-Level Optimization

### Quantization

Quantization is the most widely deployed compression technique for LLM inference. The survey distinguishes two fundamentally different workflows based on which inference stage they target.

**Weight-Only Quantization (W4A16, W3A16)**

Targets the **decoding stage** (memory-bound). Weights are stored in low precision (INT4/INT3) and dequantized to FP16 on-the-fly for computation. This reduces memory bandwidth requirements without changing the compute datatype.

| Method | Approach | Key Innovation |
|--------|----------|----------------|
| GPTQ | Layer-wise OBS with Hessian | Cholesky-based efficient inverse Hessian |
| AWQ | Activation-aware scaling | Protects salient weight channels identified by activation magnitude |
| SqueezeLLM | Non-uniform + sparse outliers | k-means clustering for centroids, sparse matrix for outliers |
| QuIP | Incoherence processing | Random orthogonal rotation before rounding |
| SpQR | Mixed precision | Identifies and preserves outlier weights at higher precision |
| OWQ | Outlier-aware columns | Allocates more bits to activation-outlier columns |
| LUT-GEMM | Lookup-table GEMM | BCQ format with precomputed partial sums |

**Weight-Activation Quantization (W8A8, W4A4)**

Targets the **prefilling stage** (compute-bound). Both weights and activations are quantized, enabling INT8/INT4 Tensor Core operations that deliver higher throughput.

| Method | Approach | Key Innovation |
|--------|----------|----------------|
| LLM.int8() | Mixed-precision decomposition | FP16 for outlier dimensions, INT8 for rest |
| SmoothQuant | Migration of quantization difficulty | Per-channel scaling shifts outliers from activations to weights |
| ZeroQuant | Token-wise + group-wise | Fine-grained activation quant, coarser weight quant |
| RPTQ | Channel reordering | Clusters channels by range, quantizes per-cluster |
| OmniQuant | Learnable clipping + migration | Optimizes clipping bounds and smoothing factors jointly |
| Atom | Mixed-precision + reordering | Isolates outlier channels, quantizes rest uniformly |

**Quantization-Aware Training (QAT)**

| Method | Approach |
|--------|----------|
| LLM-QAT | Data-free: generates training data from pre-trained model |
| QLoRA | 4-bit NF4 base + LoRA adapters in FP16 |
| QA-LoRA | Group-wise quant operators integrated into LoRA |
| LoftQ | Alternating SVD initialization for LoRA on quantized backbone |
| Norm Tweaking | Calibrate only LayerNorm parameters post-quantization |

**Critical experimental finding from the paper (Table 4, AWQ W4A16 on A100):**

| Model | Batch | Input Len | Prefill Speedup | Decode Speedup |
|-------|-------|-----------|-----------------|----------------|
| `LLaMA-2-7B` | 1 | 128 | 0.90x (slower) | 1.80x |
| `LLaMA-2-7B` | 1 | 2048 | 0.86x (slower) | 1.42x |
| `LLaMA-2-13B` | 1 | 128 | 0.95x | 2.50x |
| `LLaMA-2-13B` | 1 | 2048 | 0.87x (slower) | 1.68x |
| `LLaMA-2-7B` | 32 | 128 | 0.72x (slower) | 1.22x |
| `LLaMA-2-13B` | 32 | 128 | 0.87x (slower) | 1.40x |

Key observations:
- Weight-only quantization **increases prefilling latency** due to dequantization overhead
- Decoding speedup is substantial (1.4-2.5x) but diminishes with larger batches and longer inputs
- Larger models benefit more (13B > 7B) because they are more memory-bound

### Weight Pruning and Sparsification

**Unstructured Pruning** removes individual weights, requiring sparse matrix support for acceleration.

| Method | Pruning Criterion | Key Feature |
|--------|-------------------|-------------|
| SparseGPT | OBS (optimal brain surgeon) | Layer-wise, updates remaining weights to compensate |
| Wanda | magnitude * input activation norm | No weight update needed, extremely fast |
| RIA | relative importance + activations | Combines weight and activation information |
| Pruner-Zero | Auto-evolved symbolic metric | Uses LLM to discover pruning formulas |

**Structured Pruning** removes entire neurons, heads, or layers for hardware-friendly speedup without sparse kernels.

| Method | Structure Removed | Approach |
|--------|-------------------|----------|
| LLM-Pruner | Coupled structures (groups) | Dependency-graph-based group identification |
| Sheared LLaMA | Targeted architecture | Prunes to match a predefined smaller config |
| SliceGPT | Rows/columns via projection | Replaces pruning with orthogonal projection |
| ZipLM | Mixed structures | Iterative, guided by layer-wise speedup measurement |
| LoRAPrune | LoRA-guided | Uses LoRA gradient as importance criterion |

### Sparse Attention Patterns

For detailed coverage of attention mechanism variants (MHA, GQA, MQA, MLA, sliding window), see [A Visual Guide to Attention Variants](a-visual-guide-to-attention-variants-in-modern-llms.md). This section focuses on sparsification strategies for reducing attention cost.

**Static Patterns** define fixed sparse masks at architecture design time:

- **Sparse Transformer**: Combines local (banded) and global (strided) patterns
- **StreamingLLM**: Preserves "attention sinks" (first few tokens) plus a sliding window; enables infinite-length streaming without recomputation
- **Longformer**: Local + dilated + global attention on selected tokens
- **BigBird**: Random + local + global; provably Turing-complete

**Dynamic Token Pruning** removes unimportant tokens from the KV cache during inference:

- **Spatten**: Prunes tokens with low cumulative attention scores across layers
- **SeqBoat**: Learns per-head gating between full and linear attention
- **H2O (Heavy-Hitter Oracle)**: Retains only tokens with high accumulated attention mass plus recent tokens; reduces KV cache by 80% with minimal quality loss

**Dynamic Attention Pruning** computes attention only for relevant key-value pairs:

- **Reformer**: Locality-sensitive hashing (LSH) to bucket similar queries/keys
- **Routing Transformer**: k-means clustering of queries/keys, attend within clusters

### Low-Rank Factorization and NAS

**Low-Rank Factorization (LRF)** decomposes weight matrices W (m x n) into products of smaller matrices (m x r)(r x n) where r << min(m, n).

| Method | Target | Technique |
|--------|--------|-----------|
| LoRD | Full model | SVD, compresses `LLaMA-2-16B` to 12.3B parameters |
| TensorGPT | Embedding layer | Tensor-Train decomposition for vocabulary embeddings |
| ASVD | Attention/FFN | Activation-aware SVD (scales by activation statistics before SVD) |
| SVD-LLM | Full model | Truncation-aware data whitening + layer-wise closed-form update |
| LoSparse | Full model | Combines low-rank component with sparse residual |

**Neural Architecture Search (NAS)** finds efficient sub-architectures:

- **AutoTinyBERT**: Searches width/depth/attention structure with one-shot supernet
- **NAS-BERT**: Block-wise search with progressive shrinking
- **LiteTransformerSearch**: Training-free proxy metrics for fast evaluation

### Knowledge Distillation

**White-box KD** (access to teacher logits/hidden states):

| Method | Innovation |
|--------|-----------|
| MiniLLM | Reverse KL divergence -- avoids student overestimating low-probability teacher regions |
| GKD | On-policy: student generates, teacher scores -- matches deployment distribution |
| TED | Task-aware layer-wise distillation, decomposes by transformer components |
| MiniMoE | Distills dense teacher into MoE student for capacity-efficient compression |

**Black-box KD** (API access only, uses teacher-generated text):

| Method | Innovation |
|--------|-----------|
| Distilling Step-by-Step | Extracts rationales as additional training signal |
| Fine-tune-CoT | Teacher generates chain-of-thought, student learns to replicate |
| LaMini-LM | 2.58M diverse instructions generated by `GPT-3.5`, trains small LMs |
| Lion | Adversarial: identifies hard instructions where student fails, asks teacher for better data |

### Dynamic Inference and Early Exit

These methods adapt computation per input, exiting early from the network for "easy" inputs.

**Sample-level early exit** decides once per input whether to stop at a given layer:

- **FastBERT/DeeBERT/PABEE**: Attach classifiers at intermediate layers, exit when confidence exceeds threshold
- **HASHEE**: Hashing-based routing to predetermined exit layers

**Token-level early exit** makes per-token, per-layer decisions during generation:

- **CALM (Confident Adaptive Language Modeling)**: Each token exits at the earliest layer where a learned confidence measure exceeds a calibrated threshold. Different tokens use different amounts of computation.
- **SkipDecode**: Enforces **monotonically non-increasing** exit positions across tokens in a batch, enabling efficient batched inference without padding/masking overhead.

### Efficient Architecture Design

For detailed coverage of MoE mechanisms, see [The Big LLM Architecture Comparison](the-big-llm-architecture-comparison.md) and [From GPT-2 to gpt-oss](from-gpt-2-to-gpt-oss-analyzing-the-architectural-advances.md). For SSM/Mamba/RWKV, see [Beyond Standard LLMs](beyond-standard-llms.md).

**Key efficiency-relevant facts not covered elsewhere:**

The paper provides a direct computational complexity comparison across architecture families:

| Model | Training | Prefilling | Decoding (per token) |
|-------|----------|------------|----------------------|
| Transformer | O(n^2 * d) | O(n^2 * d) | O(n * d) |
| S4 | O(n * d^2 * log n) | O(n * d^2) | O(d^2) |
| Mamba | O(n * d^2 * log n) | O(n * d^2) | O(d^2) |
| Hyena | O(n * d * log n) | O(n * d * log n) | O(n * d * log n) |
| RetNet | O(n^2 * d) | O(n * d^2) | O(d^2) |
| RWKV | O(n * d^2) | O(n * d^2) | O(d^2) |

Critical insight: SSM/RetNet/RWKV achieve **O(d^2) per-token decoding** (constant in sequence length) vs. Transformer's O(n * d) which grows with context. However, their prefilling cost O(n * d^2) is worse than Transformer's O(n^2 * d) when n < d (typical for current models where d = 4096-8192).

Hyena is unique in having **no recurrent state** for decoding, meaning its per-token cost still depends on sequence length.

---

## System-Level Optimization

### Operator and Graph Optimization

**FlashAttention** fuses the entire multi-head attention computation (Q*K^T, masking, softmax, *V) into a single GPU kernel, eliminating materialization of the n x n attention matrix. Uses tiling to keep working set in SRAM. See [A Visual Guide to Attention Variants](a-visual-guide-to-attention-variants-in-modern-llms.md) for the attention mechanisms it accelerates.

**FlashDecoding** addresses a different bottleneck: during decoding, the query is a single token but keys/values span the full sequence. Standard FlashAttention parallelizes over batch and heads but not sequence length, leaving GPU cores idle. FlashDecoding adds a **split-K** approach, parallelizing along the KV sequence dimension.

**FlashDecoding++** further optimizes by:
1. Pre-computing a flat upper bound for softmax scaling, eliminating the synchronization barrier between split-K blocks
2. Using **FlatGEMM** kernels optimized for the (1 x n) * (n x d) shape common in decoding
3. Reducing a full transformer block to **7 fused kernels** (graph-level optimization)
4. Achieving up to **4.86x speedup** over HuggingFace Transformers

**Specialized Linear Operators** for decoding:

| System | Technique | Target |
|--------|-----------|--------|
| TensorRT-LLM | Dedicated GEMV kernels | Single-batch matrix-vector multiply |
| FlashDecoding++ | FlatGEMM | Small-batch GEMM with M=1 |
| MegaBlocks | Block-sparse GEMM | MoE FFN with variable-size experts |

### Speculative Decoding

**Core principle**: A small, fast "draft" model generates candidate token sequences, then the large "target" model verifies them in a single parallel forward pass. Mathematically guarantees identical output distribution to standard decoding (rejection sampling).

The paper provides empirical comparisons across speculative decoding variants:

| Method | Draft Source | Acceptance Rate | End-to-End Speedup |
|--------|-------------|-----------------|---------------------|
| SpD (original) | Smaller LM | 1.77-2.02x | 1.05-1.77x |
| LADE | LLM + N-gram lookup | 1.92-2.14x | 1.12-1.30x |
| SSD (self-speculative) | Sub-layers of target LLM | 1.64-1.74x | 1.01-1.23x |
| REST (retrieval) | Token datastore | 2.18-2.31x | 1.72-2.27x |
| Medusa | Extra LM heads on target | 2.52-2.62x | 2.04-2.86x |
| Eagle | Single Transformer layer | 3.47-3.72x | 2.77-3.74x |

**Eagle** dominates through three innovations:
1. **Auto-regressive draft head**: Unlike Medusa's independent heads, Eagle's draft layer predicts tokens sequentially, capturing inter-token dependencies
2. **Rich feature input**: Concatenates the target LLM's hidden states with token embeddings as input to the draft layer
3. **Token tree verification**: Organizes multiple draft sequences into a tree structure, verifying all branches in one forward pass

**Key insight**: End-to-end speedup is always lower than acceptance rate because draft model generation itself costs time. The gap is smallest for methods with negligible draft cost (Medusa, Eagle) and largest for those using separate models (SpD, SSD).

**Token tree verification** is broadly effective across all methods -- it converts a linear draft sequence into a tree of candidates, substantially increasing the expected number of accepted tokens per verification step.

### Offloading Strategies

For when model size exceeds available GPU memory:

| System | Strategy | Optimization |
|--------|----------|--------------|
| FlexGen | Offload weights/activations/KV to CPU/disk | Graph traversal to find optimal offloading schedule maximizing throughput |
| llama.cpp | Partial CPU execution | Assigns some layers entirely to CPU compute |
| PowerInfer | Hot/cold neuron splitting | GPU handles frequently-activated neurons, CPU handles rare ones (exploits ReLU sparsity: <10% neurons active) |
| FastDecode | Offload attention to CPU | Heterogeneous: GPU computes FFN, CPU handles (now-small) attention computation |

**PowerInfer** is particularly notable: it observes that with ReLU activations, fewer than 10% of FFN neurons activate for any given input. "Hot" neurons (frequently activated across inputs) are pinned to GPU; "cold" neurons (rarely activated) stay on CPU. This achieves 7-11x speedup over llama.cpp on consumer hardware.

### Serving Systems

**Memory Management for KV Cache**

The KV cache is the dominant memory consumer during serving (can exceed model weights for long sequences or large batches).

- **vLLM PagedAttention**: Borrows the operating system's virtual memory concept. KV cache is allocated in fixed-size "pages" (blocks), with a page table mapping logical positions to physical memory. Eliminates fragmentation and enables memory sharing across requests (e.g., shared system prompts). Achieves near-zero waste vs. pre-allocated contiguous buffers.
- **LightLLM**: Token-level granularity (finer than vLLM's block-level), further reducing waste at the cost of more complex indexing.

**Continuous Batching**

Traditional static batching pads all sequences to max length and waits for all to finish. Continuous batching inserts new requests as soon as any sequence completes.

| System | Granularity | Innovation |
|--------|-------------|-----------|
| ORCA | Iteration-level | First continuous batching system; selective batching per operation |
| vLLM | Iteration-level | Combined with PagedAttention |
| Sarathi / DeepSpeed-FastGen | Chunk-level | Split-and-fuse: chunks long prefills, interleaves with decode steps to maintain consistent batch utilization |

**Scheduling Policies**

| Policy | System | Behavior |
|--------|--------|----------|
| FCFS | ORCA, vLLM | Process requests in arrival order |
| Decode-prioritized | DeepSpeed-FastGen | Prioritize decode iterations to reduce latency |
| Preemptive MLFQ | FastServe | Multi-level feedback queue; preempts long requests for short ones (reduces head-of-line blocking) |
| VTC (Virtual Token Counter) | - | Fairness-oriented: ensures proportional throughput across concurrent requests |

**Disaggregated Serving**

A key architectural insight: prefilling and decoding have fundamentally different hardware requirements (compute-bound vs. memory-bound). Running them on the same machine forces compromise.

- **Splitwise / DistServe**: Separate prefill machines (high-compute GPUs) from decode machines (high-memory-bandwidth GPUs), connected by fast interconnect for KV cache transfer
- **SpotServe**: Uses preemptible cloud instances with graceful migration
- **Infinite-LLM**: Disaggregates KV cache storage across a distributed memory pool, enabling sequence-level parallelism

---

## Experimental Comparisons

**Inference Framework Throughput** (`LLaMA-2-7B` on A100):

| Framework | Inference (tok/s) | Serving (req/s) | Notes |
|-----------|-------------------|-----------------|-------|
| HuggingFace | 38.96 | - | Baseline |
| DeepSpeed | 80.95 | 6.78 | 2.1x inference over HF |
| vLLM | 90.05 | 7.11 | PagedAttention, continuous batching |
| OpenPPL | 81.17 | - | Custom CUDA kernels |
| FlashDecoding++ | 106.64 | - | Best raw inference speed |
| LightLLM | 73.60 | 10.29 | Best serving throughput |
| TensorRT-LLM | 92.51 | 5.87 | NVIDIA's optimized runtime |

Key observations:
- **FlashDecoding++** leads in raw token throughput (2.7x over HuggingFace)
- **LightLLM** leads in serving throughput (req/s) due to fine-grained memory management
- **vLLM** provides the best balance of inference speed and serving throughput
- Raw inference speed and serving throughput are **not perfectly correlated** -- serving overhead (scheduling, memory management) matters

---

## Application Scenarios and Future Directions

**Agent and Multi-Model Frameworks**: Agentic workloads (tool-use, multi-step reasoning) generate many short LLM calls in dependency graphs. Optimization opportunities include KV cache sharing across calls (SGLang), speculative execution of likely tool results, and pipeline parallelism across agents.

**Long-Context LLMs (128K-1M tokens)**: The quadratic attention cost makes brute-force infeasible. The paper identifies sub-quadratic architectures (SSM hybrids), sparse attention with hardware-aware patterns, and disaggregated KV cache as the most promising paths. See [Recent Developments in LLM Architectures](recent-developments-in-llm-architectures-kv-sharing-mhc-and-compressed-attention.md) for cross-layer KV sharing approaches.

**Edge Deployment (1-3B models)**: Models like `MiniCPM` and `MobileLLM` target mobile/embedded. Key enablers: aggressive quantization (W4A4), structured pruning, and specialized runtimes (MLC-LLM). The primary constraint shifts from throughput to memory footprint and power consumption.

**Security-Efficiency Tradeoffs**: The paper raises an underexplored concern -- aggressive quantization (below 4 bits) may compromise emergent abilities (chain-of-thought, in-context learning) disproportionately relative to perplexity degradation. Safety alignment may also be fragile to compression.

---

## Key Takeaways

1. **Match technique to bottleneck**: Weight-only quantization for decoding (memory-bound), weight-activation quantization for prefilling (compute-bound). Applying the wrong technique can actually hurt performance.

2. **Speculative decoding is the highest-leverage system optimization**: Eagle achieves 2.77-3.74x speedup with no quality loss. The field is converging on auto-regressive draft heads with token tree verification.

3. **Quantization has asymmetric effects**: W4A16 delivers 1.4-2.5x decode speedup but **slows prefilling by 10-28%** due to dequantization overhead. This is often overlooked in benchmark reports that focus only on throughput.

4. **KV cache management is as important as model compression**: vLLM's PagedAttention and continuous batching often provide larger serving throughput gains than model-level optimizations alone.

5. **Disaggregated serving is the emerging paradigm**: Separating prefill and decode onto different hardware matches their fundamentally different resource profiles and is likely to become standard for large-scale deployments.

6. **Sub-quadratic architectures trade prefill cost for decode cost**: SSM/RWKV/RetNet achieve O(d^2) per-token decode (vs. Transformer's O(n*d)) but O(n*d^2) prefill (worse when n < d). The crossover point matters.

7. **Practical deployment requires combining techniques across all three levels**: The best serving systems (vLLM, TensorRT-LLM) already combine operator fusion, quantization, paged memory, and continuous batching. Future systems will add speculative decoding and disaggregated compute.

8. **Edge deployment needs fundamentally different tradeoffs**: Memory footprint and power, not throughput, are the binding constraints. Structured pruning and W4A4 quantization matter more than batching optimizations.

---

## References

- Zhou, Z., Ning, X., Hong, K., et al. "A Survey on Efficient Inference for Large Language Models." arXiv:2404.14294v3, July 2024.
- Kwon, W., et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023.
- Dao, T. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." 2023.
- Li, Y., et al. "Eagle: Speculative Sampling Requires Rethinking Feature Uncertainty." ICML 2024.
- Lin, J., et al. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." MLSys 2024.
- Frantar, E., et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." ICLR 2023.
- Xiao, G., et al. "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." ICML 2023.
- Frantar, E. and Alistarh, D. "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot." ICML 2023.
- Sun, M., et al. "Wanda: A Simple and Effective Pruning Approach for Large Language Models." 2024.
- Hong, K., et al. "FlashDecoding++: Faster Large Language Model Inference on GPUs." 2024.
- Leviathan, Y., et al. "Fast Inference from Transformers via Speculative Decoding." ICML 2023.
- Schuster, T., et al. "Confident Adaptive Language Modeling." NeurIPS 2022.

---

*Cross-references: [A Visual Guide to Attention Variants](a-visual-guide-to-attention-variants-in-modern-llms.md) | [The Big LLM Architecture Comparison](the-big-llm-architecture-comparison.md) | [Beyond Standard LLMs](beyond-standard-llms.md) | [From GPT-2 to gpt-oss](from-gpt-2-to-gpt-oss-analyzing-the-architectural-advances.md) | [Recent Developments in LLM Architectures](recent-developments-in-llm-architectures-kv-sharing-mhc-and-compressed-attention.md)*
