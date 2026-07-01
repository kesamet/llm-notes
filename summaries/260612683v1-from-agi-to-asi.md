# From AGI to ASI

> Based on Genewein et al.'s report (June 2026)
> Source: https://arxiv.org/abs/2606.12683

---

## Table of Contents

- [Overview](#overview)
- [Characterizing AGI, ASI, and Universal AI](#characterizing-agi-asi-and-universal-ai)
- [Working definitions of AGI, ASI, and UAI](#working-definitions-of-agi-asi-and-uai)
- [Advantages of Digital Intelligence](#advantages-of-digital-intelligence)
- [Fundamental Limits of ASI](#fundamental-limits-of-asi)
- [Universal AI (AIXI) -- An Informal Overview](#universal-ai-aixi--an-informal-overview)
- [Technological Pathways from AGI to ASI](#technological-pathways-from-agi-to-asi)
- [1. Scaling Compute, Models, and Data](#1-scaling-compute-models-and-data)
- [2. Algorithmic Paradigm Shifts and Evolutions](#2-algorithmic-paradigm-shifts-and-evolutions)
- [3. Recursive Self-Improvement](#3-recursive-self-improvement)
- [4. Multi-Agent Coordination and Group Agency](#4-multi-agent-coordination-and-group-agency)
- [Potential Bottlenecks and Frictions](#potential-bottlenecks-and-frictions)
- [Remarks](#remarks)
- [Is Quantitative Compute Scaling Enough?](#is-quantitative-compute-scaling-enough)
- [Predicting What ASI Can and Cannot Do](#predicting-what-asi-can-and-cannot-do)
- [Is Superintelligence Super-Creative?](#is-superintelligence-super-creative)
- [What Goals Might ASI Pursue?](#what-goals-might-asi-pursue)
- [Does AGI Have to Be Agentic?](#does-agi-have-to-be-agentic)
- [Research Agenda](#research-agenda)
- [Key Takeaways](#key-takeaways)
- [References](#references)

---

## Overview

This Google DeepMind report maps the landscape of AI progress *beyond* human-level AGI, asking how intelligence might continue to develop toward **artificial general superintelligence (ASI)** -- intuitively, systems more capable than large, well-coordinated organizations of human experts. Rather than predict a date for AGI, it characterizes ASI, grounds it in the theoretical limit of **Universal AI (AIXI)**, and surveys four technological pathways and the frictions that could slow each.

The central framing is a tension between accelerating and decelerating exponential dynamics that "race against each other." Growth in **effective compute** (≈10x per year) and potential **recursive self-improvement** push capabilities up, while diminishing returns, resource limits, and research difficulty push back. The report argues the outcome of this race is genuinely uncertain, so the appropriate response is a massively interdisciplinary research program of forecasting, benchmarking, and theoretical work.

A key caveat runs throughout: even if progress continues far past human level, ASI is **not omniscient or omnipotent** -- it is bound by well-understood physical, complexity-theoretic, and logical limits. The image of a single transformative "step change" from AGI may be inaccurate; more likely is a series of transformative societal changes driven by AI-enabled breakthroughs across science and technology.

---

## Characterizing AGI, ASI, and Universal AI

The report uses **qualitative, coarse** characterizations grounded in the **Legg-Hutter score** -- a formal measure of intelligence as the average performance of an agent across all computable tasks (weighted by inverse Kolmogorov complexity). Because this defines a *continuum* of intelligence, the report avoids needing a sharp threshold between AGI and ASI; what matters is a significant score gap.

### Working definitions of AGI, ASI, and UAI

| Term | Definition | Calibration |
|---|---|---|
| **AGI** | Human-level artificial general intelligence; roughly median human-level on most cognitive tasks ("Competent AGI" in Morris et al. 2024) | Comparable to a single median human; already superhuman on many narrow tasks |
| **ASI** | Artificial general superintelligence; superhuman across virtually all tasks and domains | Exceeds the performance of large, well-coordinated expert collectives (tens of thousands of experts over ~10 years, with 2010-era technology) |
| **UAI** | Universal AI; the theoretical limit of superintelligence, formalized via the AIXI agent | Incomputable endpoint on the continuum; approximable only "from below" with more powerful ASIs |

Narrow superhuman systems like `AlphaFold` or `AlphaGo` are explicitly ruled out as ASI because they excel only in single domains. A single ASI may itself be a collective of millions of interacting instances. The human-performance baseline is a **moving target** (humans get more capable with better technology), which is why ASI is placed at a deliberate step change above today's median individual.

### Advantages of Digital Intelligence

The most distinctive property of AI is that we know its full algorithmic description (its code), implying **substrate independence** -- the same AI runs on any sufficiently powerful computer. This fact propagates into a set of advantages over biological intelligence that *intensify with more compute*.

| Advantage | How it scales with compute |
|---|---|
| **I/O speed** | AI ingests information and produces outputs at ever-higher bandwidth (today's LLMs ingest multiple books in seconds) |
| **Internal processing speed** | "Thinking" speeds up via faster sequential (depth) or parallel (breadth) computation; major scaling advantage even under diminishing returns |
| **Working memory & memorization** | Dramatically larger capacity and read/write bandwidth; memorizing large parts of the internet is far from the ceiling |
| **Substrate independence** | Can migrate between computers, even at runtime or component-by-component onto heterogeneous hardware |
| **Lossless replication** | Copyable including memory state ("lifetime experience"); enables backup/restore and spawn/halt/resume on demand |
| **High-bandwidth experience sharing** | I/O streams can be stored, shared, and replayed; homogeneous instances can share raw learning signal (e.g., averaged gradients) |

A notable counterpoint (from Lawrence 2024): humans' low I/O bandwidth ("high embodiment factor") *forces* them to form deep internal models and coarse abstractions. Machines with high-bandwidth I/O may never *need* such abstractions, raising the open question of whether digital intelligence forms human-like abstractions at all (see the **Abstraction Barrier** below).

### Fundamental Limits of ASI

ASI is neither all-knowing nor all-powerful. Several of its limits are precisely characterizable through the AIXI framework.

| Limitation | Nature |
|---|---|
| **Fundamental physics** | Speed of light (info propagation), Landauer principle (energy per erasure), Bremermann's limit (max computation speed), Bekenstein bound (max info in finite space/energy) |
| **Real time** | The physical world runs in real time; experiments that resist precise simulation (weather, organisms, economies, societies) are bound by it |
| **Physical manipulation** | Not all logically possible matter configurations are physically realizable; building takes time and energy (cf. Universal Constructor) |
| **Ignorance, observability & controllability** | Epistemic uncertainty and finite measurement precision impose limits on predictability and controllability |
| **Complexity theory** | P vs. NP vs. PSPACE bounds apply, though worst-case limits often exceed practical (approximate) performance |
| **Logic** | Gödel's Incompleteness and the Halting Problem bound what can be objectively answered or known |

The crux: these limits hold exactly, but most do *not* settle whether specific capabilities (curing ageing, brain uploading, climate restoration) are possible, because good approximations can achieve strong practical performance far below worst-case bounds.

---

## Universal AI (AIXI) -- An Informal Overview

**Universal AI (UAI)**, formalized by the **AIXI** agent, is the best-understood theoretical upper bound on machine intelligence. It bounds ASI *from above*; today's empirical deep learning approaches it *from below*. The framework matters because it lets us reason about intelligence in the limit even when today's systems differ greatly.

AIXI is a general agent optimal over the class of **all computable environments** (dynamics plus a reward function). This is far broader than standard ML/RL assumptions (stationarity, ergodicity, Markovian dynamics). It solves three fundamental problems:

| Problem | How AIXI addresses it |
|---|---|
| **Acting under uncertainty** | Maintains a Bayesian posterior mixture over all computable environments as its world model; updates it with observations. Prior weights via Solomonoff's Universal Prior (lower Kolmogorov complexity = exponentially more likely) |
| **Interactive decision-making** | General RL over arbitrary computable dynamics/rewards; optimizes long-horizon cumulative reward via planning |
| **Exploration-exploitation** | Solved implicitly -- high initial uncertainty makes information-gathering actions high expected reward; exploration naturally tapers as certainty grows |

AIXI maximizes expected cumulative reward averaged over all computable environments weighted by the universal prior. It is Pareto-optimal: no agent achieves higher *expected* reward under this prior, though it may underperform in specific environments. It inherits Solomonoff Induction's guarantee of being the most data-efficient learner on average (lowest cumulative prediction error).

The big catch: **AIXI is incomputable**. Practical algorithms approximate it from below and improve with more compute/data, but brute-force approximations need rapidly growing compute for linear intelligence gains. Recent theoretical moves close the gap to practice:

- Most of AIXI's heavy lifting can be pushed into the predictor part (Catt et al. 2023; Kim and Lee 2026).
- Training an amortized Bayesian predictor via log-loss minimization with a large parametric model can in principle reach the universal limit (Grau-Moya et al. 2024); pretraining a massive sequential predictor to minimize log-loss over internet-scale data is a resource-bounded approximation of universal compression that improves with scale (Genewein et al. 2026).
- The AIXI "recipe" then suggests layering explicit planning/decision-making scaffolding (test-time search) on top of the universal predictor.

This lends theoretical support to the conjecture that the modern pretraining + fine-tuning paradigm can be pushed into ASI territory *without fundamental theoretical blockers* -- assuming sufficient model expressivity and optimizers. Known shortcomings of the current paradigm (continual learning, very long context, robust planning) remain, and practical ASI may be built before the theory is unified.

Alternative/complementary theoretical frameworks motivated by AIXI's limits: reflective oracles, logical induction, Schmidhuber's Gödel machines, computational mechanics, PAC-learning, algorithmic game theory, and thermodynamic bounded rationality (Landauer-based energy bounds on intelligence).

---

## Technological Pathways from AGI to ASI

The report identifies four largely independent, potentially parallel pathways. Only the first has historic data to fit forecasting models against.

| Pathway | Main uncertainty |
|---|---|
| Scaling compute, models & data | How scale increases translate into capabilities (spiky vs. smooth; emergent capabilities; diminishing returns) |
| Algorithmic paradigm shifts | High unpredictability of progress; novel-paradigm frictions |
| Recursive (self-) improvement | Dynamics unclear and without historic precedent; could explode (hyperbolic) or taper quickly |
| ASI via group agent formation | Whether ASI emerges from orchestration or self-organizing market dynamics; emergence in complex systems is poorly understood |

### 1. Scaling Compute, Models, and Data

Recent AI progress came from scaling models, data, and compute. The case for scaling rests on the **bitter lesson** (Sutton 2019): if intelligence is search (learning = search through hypothesis space; planning = search through policy space), more compute means more search and thus more intelligence.

The complication is that **naive brute-force search fails in virtually all non-toy domains**, including chess. Capability gains come from *search efficiency* -- better priors, inductive biases, surrogate models, and parametric value estimators. This makes the practical compute-to-intelligence relationship non-obvious. Performance has scaled predictably along approximate power laws in parameters, data, and compute (Kaplan et al. 2020), but it is open whether quantitative scaling alone reaches ASI or whether qualitative paradigm shifts are also required.

A near-future friction is the **data wall**: high-quality text exhaustion is estimated later this decade (Villalobos et al. 2024). Recent corpora reach ~3 trillion tokens via filtering/deduplication. Bridging AGI-to-ASI likely requires transcending human-generated data limits, even counting non-text modalities. Architectural innovations like sparse **Mixture-of-Experts** extend the scaling runway by raising compute efficiency (trillion-parameter regimes with manageable footprints). MoE mechanisms are covered in detail in [The Big LLM Architecture Comparison](the-big-llm-architecture-comparison.md).

The central question is not so much "is scaling enough?" but **"can scaling be sustained long enough"**, given that economic inputs and natural resources must also scale through many orders of magnitude.

### 2. Algorithmic Paradigm Shifts and Evolutions

The current paradigm -- supervised pretraining of large transformers via log-loss, then fine-tuning, then test-time scaling, retrieval, and tool use -- has broad consensus as *insufficient for AGI*. The community is adding "missing ingredients." The report distinguishes **evolutions** (active research, frontier-scale) from **paradigm shifts** (dramatic, hard-to-anticipate changes, likely arising from hitting the paradigm's ceiling).

Concrete evolutions discussed, several of which overlap with existing KB pages:

| Evolution | What it adds | KB coverage |
|---|---|---|
| Test-time scaling | Decouples intelligence from static training constraints via dynamic adaptive computation, tool-augmented planning | Covered in [Beyond Standard LLMs](beyond-standard-llms.md) and [A Survey on Efficient Inference](a-survey-on-efficient-inference-for-large-language-models.md) |
| Continual learning | Perpetually accrues competence without catastrophic forgetting | -- |
| Unbounded context | Retrieval systems for virtually infinite, updateable working memory; linear-time sequence architectures (`Mamba`, `S4`) to remove quadratic attention bottlenecks | Linear attention in [Beyond Standard LLMs](beyond-standard-llms.md); retrieval in [A Survey on Efficient Inference](a-survey-on-efficient-inference-for-large-language-models.md) |
| Internal world models | Compressed, manipulable representations of dynamics for simulation, long-horizon planning, counterfactual reasoning | Code World Models in [Beyond Standard LLMs](beyond-standard-llms.md) |

True paradigm shifts (spiking neurons / neuromorphic or analog hardware, RL-based pretraining, explicit world models, overcoming complexity-theoretic limits) are near-impossible to predict, making this pathway least amenable to forecasting -- but it should not be dismissed. Advancing paradigm-agnostic fundamental understanding of superintelligence and its bounds is the main lever for reducing this uncertainty.

### 3. Recursive Self-Improvement

**Recursive (self-) improvement** is AI facilitating AI R&D, producing better AI, which accelerates R&D further -- potentially an "explosive" AGI-to-ASI transition if it becomes fully autonomous. Beyond the classic notion of AI writing better code, the report identifies four flavors mapped onto human evolutionary processes:

| RSI flavor | Human analogue | AI form |
|---|---|---|
| **Genotypic** (code/hardware) | Genetic evolution (slow) | AI modifying its "DNA" -- architectures, optimizers, chip blueprints, manufacturing -- potentially rapidly and targeted |
| **Memetic** (data) | Cultural evolution (fast, last 50k years) | Automated dataset collection/curation, synthetic data, recursive distillation of test-time search (AlphaZero-style) |
| **Sociogenic** (cooperation) | Division of labor | Specialization freeing resources for more instances and further specialization |
| Algorithmic self-improvement | -- | LLM-guided program search (`FunSearch`, `AlphaEvolve`) finding solutions beyond training distribution |

Concrete weak recursive loops are arguably already at play: neural architecture search, automated hyperparameter tuning, AI-assisted hardware design (`ChipNeMo`, Mirhoseini et al. 2021), auto-curricula, and simulations with learned world models. Formal barriers exist (Schmidhuber's Gödel machines require complete self-knowledge; Christiano's iterated amplification offers a more practical bootstrapping approach). "AI Scientist" systems (`Lu et al. 2024`, `AlphaEvolve`, Kosmos) show more autonomous improvement may be near.

Whether it fizzles or explodes is unclear. Key dampeners: even digital researchers must run larger experiments and wait for outcomes (especially physical-world experiments), and resource consumption can explode. The report's main recommendation here is to seek **"recursive improvement scaling laws"** predicting self-improvement curves from early-onset data.

### 4. Multi-Agent Coordination and Group Agency

ASI may emerge as a **collective property** of many coordinated AGI agents, analogous to how human intelligence aggregates into superintelligent institutions. Drawing on group agency theory, AGIs could form coherent "Group Agents" (e.g., fully automated corporations) with motivational states distinct from their constituents. This is the topic of the existing KB page [Intelligent AI Delegation](2602.11865v1.Intelligent_AI_Delegation.md), directly cited here as Tomašev et al. 2026.

Two organizing poles:

| Form | Mechanism | Analogy |
|---|---|---|
| **Decentralized** | Local incentives aggregate into higher-order intelligence via price signals in "Virtual Agent Economies"; capabilities surpass any individual participant | Human financial markets |
| **Centralized** | High-bandwidth communication lets a "CEO" literally talk to every employee/voter, reducing hierarchy depth and bureaucratic friction | Cybernetic collective |

Collective intelligence may scale with agent population size and interaction density (conditioned on compute), motivating the search for **"Multi-Agent Scaling Laws."** For human organizations, collective intelligence depends mainly on parallelization (overcoming individual bandwidth limits) and diversity (specialization enabling synergies homogeneous groups cannot). Whether a homogeneous LLM collective (with different prompts/contexts) yields synergistic gains is an open question -- as is how to steer AGI groups and handle intelligence/bandwidth asymmetries in mixed human-AI collectives.

---

## Potential Bottlenecks and Frictions

For each bottleneck, the report pairs a description with factors that might counteract it. Whether a bottleneck becomes fundamental or is merely a friction is treated as an **open research question**.

| Bottleneck | Description | Countered by |
|---|---|---|
| **Data wall** | Running out of sufficiently growing high-quality data for pretraining/post-training/test-time adaptation | Synthetic data, high-fidelity simulations, self-generated interaction data (RL, self-play), paradigm shifts raising data efficiency |
| **Economic & natural resource demand** | Growth in investments, chips/supply chains, energy, datacenter sites, rare earths cannot be sustained | Increasing economic returns from AI deployment; AI-driven efficiency gains; large-scale infrastructure buildout |
| **Neural paradigm insufficient** | AGI cannot be reached with large pretrained neural networks (+ post-training, test-time scaling, scaffolding, tool use) or SGD | Continued research for paradigm evolutions/shifts; even sub-AGI systems may accelerate that research |
| **Research gets harder** | Effort for continued progress rises as fields mature ("low-hanging fruit" harvested) | More capable AI improving research and resource efficiency (fewer experiments, better hypotheses) |
| **Abstraction barrier** | Systems trained on human abstractions may lack the ability to form novel concepts from raw data | Even if individuals plateau near human level, scaling + group formation could push collective capability beyond AGI; a paradigm shift (interactive learning & RL) may address it directly |
| **Deliberate slowdown** | Rogue-actor use, accidents, military/political abuse, or societal backlash could trigger regulatory caps on capability growth | Economic/political pressure and international race dynamics may override slowdown, especially without global coordination |

Two deserve particular elaboration:

- **The data wall** is nuanced. Naive iterated training on self-generated data leads to model collapse / degeneration (Shumailov et al. 2024), but test-time scaling (search improving generations, distilled back into the base model, AlphaZero-style) may produce high-quality data just beyond a base model's frontier. Simulations and RL/agent interaction data also scale with compute (e.g., DeepMind's Adaptive Agent). The upshot: data may be a *friction, not a fundamental blocker*, if progress is compute-driven.

- **The abstraction barrier** (formulated by Lerchner) hypothesizes that systems trained on human cognitive products are bounded by existing conceptual frameworks -- computation alone cannot instantiate novel conceptual primitives without an experiencing agent mapping physical reality to symbols. Illustration: a model trained on pre-Newtonian-era scientific text at modern token counts could almost certainly not reason its way to general relativity or quantum mechanics without the primitives of calculus, gravitation, or electromagnetism. If this barrier caps individual intelligence at AGI level, a path to ASI runs through **grounded concept discovery** from raw sensor data, validated against physical reality (the **Embodied Bottleneck**) -- constraining recursive hardware self-improvement to real-world experiment speeds.

The deliberate-slowdown bottleneck interacts with governance: compute-threshold licensing (EU AI Act), mandatory evaluations/incident reporting, and corporate responsible-scaling policies embed institutional gates that major scaling steps must pass. Large accidents or credible near-misses could render further scaling politically/commercially infeasible even where technically achievable. But "military-economic adaptationism" -- whereby actors adopting power-enhancing technologies are differentially selected to survive -- may override unilateral slowdowns.

---

## Remarks

### Is Quantitative Compute Scaling Enough?

In *theory*, yes: AIXI approximations improve toward Universal Intelligence with more compute, and intelligence-as-search would benefit from open-ended search. In *practice*, naive brute-force search hits resource constraints rapidly; effective search depends on inductive biases and priors that constrain the hypothesis class. Strong inductive biases help data efficiency but cap maximum general intelligence -- and that cap cannot be overcome by more compute alone, requiring qualitative innovation instead.

The big caveat: even if *individual* AI capabilities plateau near human level, **collectives** of AGI instances might become superhuman through scale. With effective compute growing ~10x per year, millions of AGI instances become runnable within a few years (MacAskill estimates AI "population scaling" at ~25x per year). The open question is for *which kinds of tasks* groups beat individuals. Test-time scaling of today's models has only limited headroom before plateauing, so it alone would not take an AGI to ASI -- but running more instances and forming groups could.

### Predicting What ASI Can and Cannot Do

Specific ASI capabilities (curing diseases, fusion, unifying relativity and QM) cannot be answered today. Extrapolating from current capabilities becomes highly uncertain fast; theoretical limits (complexity, AIXI bounds) are mostly *negative* and often vacuous, because good approximations and heuristics can achieve strong performance far below worst-case bounds. Worse, knowing *whether* good approximations exist for a problem is itself often **computationally irreducible** -- the only way to know is to find them and run them. This unpredictability is inherited from universal compression (Kolmogorov's structure function): how good lossy compressions are cannot be predicted in advance.

The practical implication: prediction needs an **empirically-first approach complemented by theory**. Scaling laws (Kaplan et al. 2020) and **benchmark stitching** (Ho et al. 2025) have worked well for same-family models. Existing benchmarks are saturating rapidly (`GPQA`, `SWE-bench`, `FrontierMath`), highlighting the need for benchmarks measuring true generalization (`ARC-AGI`) and for private/continuous adversarial evaluation.

### Is Superintelligence Super-Creative?

Using Boden's framework (a creative product must be novel, surprising, and valuable), creativity stratifies into three levels by type of surprise:

| Level | Creativity type | Example |
|---|---|---|
| 1 | **Combinational** -- unfamiliar combinations of familiar ideas | Poetic imagery, novel recombination of modules |
| 2 | **Exploratory** -- new elements within existing conceptual spaces | A new move in a known game (AlphaGo's Move 37) |
| 3 | **Transformative** -- creating entirely new conceptual spaces | Quantum theory, Cubism, inventing a new type of game |

AI achievements to date (Move 37, automated theorem proving, `AlphaFold`) belong predominantly to levels 1 and 2 -- profound exploratory creativity within human-provided conceptual spaces. Reaching Boden's level 3, transformative creativity, may be the hallmark requirement of ASI. Demis Hassabis's proposed "true test": could an AI, given only the information Einstein had in 1900, have come up with general relativity? (Today: no.) Inventing new scientific theories triggering Kuhnian paradigm shifts would satisfy the criteria.

### What Goals Might ASI Pursue?

As AI scales beyond human level, its specific final goals become hard to predict, but **instrumental convergence** (Omohundro 2008; Bostrom 2012) describes universally useful sub-goals pursued regardless of final goal: resource acquisition, time efficiency, and self-preservation (resisting shutdown to avoid goal-prevention). Preservation is a technical problem with theoretical solutions -- **Corrigibility** and "Safely Interruptible Agents" -- though translating these to frontier-scale systems remains open.

The objective matters for stability. Standard RL (maximizing scalar rewards) risks reward hacking, stagnation, or the "Delusion Box" (an agent modifying its sensory inputs for max reward). The **Knowledge Seeking (KS)** objective (Orseau 2014) instead maximizes information gain, with attractive properties: robustness to delusions, avoiding stagnation, aversion to irreversible changes, and favoring cooperation (knowledge is non-rivalrous and positive-sum).

### Does AGI Have to Be Agentic?

Not necessarily. **Oracles** can answer questions at superintelligent level without pursuing their own goals; the "Scientist AI" framework (Lu et al. 2024) proposes systems that explain observations and generate world models without taking goal-directed actions, remaining "boxed" to mitigate risk. **Myopic AI** optimizes short-horizon rewards, potentially avoiding convergent instrumental goals of resource acquisition and self-preservation.

Caveats: an oracle interacting with a persistent world *is* an agent (action space = text output) with reduced controllability; even a pure prediction-error-minimizing oracle gains implicit incentives to control the future and manipulate users toward predictable questions. The economic and practical pressure to reduce human-in-the-loop oversight remains a strong driver toward full autonomy, so the most impactful systems will likely integrate these capabilities into autonomous agents. Agent harnesses and their design tradeoffs are covered in [Components of A Coding Agent](components-of-a-coding-agent.md).

---

## Research Agenda

The report frames its open questions as a research program (Section 7.1), grouped thematically:

1. **Bottlenecks and Frictions for Scaling** -- Can data generation meet scaling demands? When is third-party experience sufficient for learning to act? When does more compute yield more intelligence? What can be anticipated about paradigm shifts? When does scaling become economically unviable? How does the Embodied Bottleneck limit intelligence growth? Does the Abstraction Barrier fundamentally bound capabilities?

2. **Quantitative Forecasting** -- Couple effective-compute growth with capability gains and macroeconomic effects (Epoch's **GATE model**; Davidson et al. 2026's explosive-growth model). Identify macro-quantities (cost per FLOP, compute efficiency, economic productivity), build coupled mathematical models with ensembling, simulate scenarios to find inflection points/thresholds, and establish continuous-update protocols.

3. **Benchmarking ASI** -- Comparing against human performance will not distinguish superhuman systems. Candidate approaches: multi-agent benchmarks (zero-sum competition, and superhuman *cooperative* benchmarks), setter-solver (AI-automated benchmark design), general compression benchmarks (Universal Induction motivated), and indirect measurements (economic productivity, resource efficiency). Also: distinguish true qualitative leaps from metric-saturation artifacts; design non-saturating, low-human-involvement ASI benchmarks.

4. **Recursive Improvement Dynamics** -- Identify mechanisms, measure effects, establish scaling laws. How far can a fixed model be pushed with test-time compute alone? Can AI meaningfully curate its own training data? Develop a theory of recursive distillation (trade-offs between base-model size and search; degeneration conditions; verifier quality). Track "AI Scientist" productivity. Whether specialization/division of labor yields significant recursive gains.

5. **Multi-Agent Scaling** -- How task delegation bypasses individual-agent limits; for which task classes groups beat individuals and how this depends on organization form (homogeneous collective vs. heterogeneous market); develop multi-agent scaling laws; whether more instances beats larger monolithic models per compute; group alignment against epistemic hijacking; epistemic resilience in asymmetric-intelligence collectives.

6. **Advance the Theoretical Foundations of Superintelligence** -- Adapt AIXI for practical ASI analysis; understand where good approximations are possible and how to predict their quality; complexity-theoretic limits of lossy compression; whether capability "jaggedness" is fundamental or a human-comparison artifact; novel frameworks for myopic/non-agentic advanced AI.

7. **AI Safety, Alignment, Sociocultural** -- How to implement deliberate slowdown (taxation vs. prohibition); what makes AIs/groups easier to robustly align; risks of convergent instrumental sub-goals; pressures on the scientific process under automated research; economic impacts of the labor-to-capital shift.

The report's bottom-line position (stated with low confidence): it is *more likely* that AI progress either plateaus *before* AGI level, or goes from AGI to (weak) ASI relatively smoothly -- assuming no dramatic recursive-self-improvement acceleration, which cannot be ruled out and would make the transition rapid. The possibility of cruising past AGI into ASI territory within the next one-to-two decades "cannot easily be dismissed."

---

## Key Takeaways

1. **ASI is defined against expert collectives, not individual humans.** The report places ASI at a deliberate step above AGI -- exceeding large, well-coordinated groups of tens of thousands of experts over ~10 years with 2010-era technology -- while grounding both notions in the Legg-Hutter intelligence continuum rather than sharp thresholds.

2. **Effective compute grows ~10x per year, compounding three factors.** Hardware (~1.5x/yr), investment (~2.5x/yr), and algorithmic efficiency (~3x/yr, possibly ~6x/yr) multiply to roughly an order of magnitude annually, but how this translates into *new* capabilities (vs. merely more/faster instances) is the core uncertainty.

3. **Universal AI (AIXI) bounds ASI from above and is surprisingly practical as intuition.** It is incomputable but approximable from below, and recent results argue the modern pretraining-as-compression paradigm can in principle be pushed toward universal intelligence without fundamental theoretical blockers.

4. **The four pathways are complementary, not competing.** Scaling, paradigm shifts, recursive self-improvement, and multi-agent group formation can proceed in parallel, and their gains can compound; only scaling has historic data to extrapolate from.

5. **Recursive self-improvement spans more than code.** Beyond AI writing better architectures, the report catalogs hardware, data/memetic (AlphaZero-style distillation), and specialization/sociogenic flavors -- each mapped onto a human evolutionary analogue.

6. **Every bottleneck is paired with a counter, and each pairing is an open question.** The data wall, resource demand, paradigm limits, research difficulty, the abstraction barrier, and deliberate slowdown could each be a hard blocker or merely a friction depending on how effective its counters prove at scale.

7. **The abstraction barrier may cap individual AI intelligence at AGI level.** If systems trained on human abstractions cannot form novel concepts from raw data, ASI may require grounded concept discovery validated against physical reality -- the Embodied Bottleneck -- which linearly slows recursive improvement to the pace of empirical science.

8. **Capabilities are often unpredictable in principle.** Because the quality of approximations is frequently computationally irreducible, predicting what ASI can do needs an empirically-first approach (scaling laws, benchmark stitching) complemented by theory; this is why forecasting and non-saturating benchmarking are framed as core research fields.

9. **Transformative (level-3) creativity may be the hallmark of ASI.** AI achievements to date sit in Boden's combinational/exploratory levels; inventing genuinely new conceptual spaces (Hassabis's "derive general relativity from 1900 information") would mark the step to transformative creativity.

10. **The dominant near-term risk is not omniscience but autonomy pressure.** Human feedback is slow and expensive, creating pressure toward autonomous, internally-objective-driven systems; objectives like Knowledge Seeking are attractive precisely because they favor cooperation and resist delusion.

---

## References

- From AGI to ASI (Genewein et al., June 2026): https://arxiv.org/abs/2606.12683
- Morris et al., Levels of AGI (2024): https://arxiv.org/abs/2311.02462
- Legg & Hutter, Universal Intelligence (2007): https://doi.org/10.1007/s11023-007-9079-x
- Hutter et al., An Introduction to Universal Artificial Intelligence (2024): http://www.hutter1.net/ai/uaibook2.htm
- Catt et al., Self-predictive universal AI (2023): https://proceedings.neurips.cc/paper_files/paper/2023
- Grau-Moya et al., Learning universal predictors (2024): https://proceedings.mlr.press/v235/grau-moya24a.html
- Genewein et al., Algorithmic compression via pretrained neural networks (2026): https://www.mdpi.com/1099-4300/28/6/596
- Kim & Lee, A model-free universal AI (2026): https://arxiv.org/abs/2602.23242
- Meulemans et al., Embedded universal predictive intelligence (2025): https://arxiv.org/abs/2511.22226
- Kaplan et al., Scaling laws for neural language models (2020): https://arxiv.org/abs/2001.08361
- Ho et al., A rosetta stone for AI benchmarks / benchmark stitching (2025): https://arxiv.org/abs/2512.00193
- Ho et al., Algorithmic progress in language models (2024): https://proceedings.neurips.cc/paper_files/paper/2024
- Villalobos et al., Will we run out of data? (2024): https://arxiv.org/abs/2211.04325
- Shumailov et al., AI models collapse when trained on recursively generated data (2024): https://www.nature.com/articles/s41586-024-07566-y
- Sutton, The bitter lesson (2019): http://www.incompleteideas.net/IncIdeas/BitterLesson.html
- Davidson et al., When does automating AI research produce explosive growth? (2026): http://www.nber.org/papers/w35155
- Erdil et al., GATE integrated assessment model (2025): https://arxiv.org/abs/2503.04941
- Chan et al., Measuring AI R&D automation (2026): https://arxiv.org/abs/2603.03992
- Bloom et al., Are ideas getting harder to find? (2020): https://www.aeaweb.org/articles?id=10.1257/aer.20180338
- Romera-Paredes et al., FunSearch / mathematical discoveries from program search (2024): https://www.nature.com/articles/s41586-023-06924-6
- Novikov et al., AlphaEvolve (2025): https://arxiv.org/abs/2506.13131
- Lu et al., The AI Scientist (2024): https://arxiv.org/abs/2408.06292
- Bengio et al., Scientist AI / superintelligent agents catastrophic risks (2025): https://arxiv.org/abs/2502.15657
- Orseau, Universal knowledge-seeking agents (2014): https://doi.org/10.1016/j.tcs.2013.09.025
- Orseau & Armstrong, Safely interruptible agents (2016): https://arxiv.org/abs/1606.06560
- Trivedi et al., Solipsistic superintelligence is unlikely to be cooperative (2026): ICML 2026 (Proceedings of Machine Learning Research)
- Tomašev et al., Intelligent AI Delegation (2026): https://arxiv.org/abs/2602.11865
- Tomašev et al., Virtual agent economies (2025): https://arxiv.org/abs/2509.10147
- Lerchner, The abstraction fallacy (2026): https://philarchive.org/rec/LERTAF
- Schrittwieser et al., MuZero / mastering Atari, Go, chess, shogi (2020): https://www.nature.com/articles/s41586-020-03051-4
- Silver et al., Mastering the game of Go without human knowledge (2017): https://www.nature.com/articles/nature24270
- Gu & Dao, Mamba (2024): https://openreview.net/forum?id=tEYskw1VY2
- Lewis et al., RAG (2020): https://proceedings.neurips.cc/paper/2020
- Chollet, On the measure of intelligence / ARC-AGI (2019): https://arxiv.org/abs/1911.01547
- Glazer et al., FrontierMath (2024): https://arxiv.org/abs/2411.04872
- MacAskill & Moorhouse, Preparing for the intelligence explosion (2025): https://arxiv.org/abs/2506.14863
- Epoch AI, Key trends and figures in machine learning: https://epoch.ai/trends

