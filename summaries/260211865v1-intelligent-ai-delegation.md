# Intelligent AI Delegation -- Wiki

> Based on Nenad Tomasev, Matija Franklin, and Simon Osindero's paper (February 2026)
> Source: https://arxiv.org/abs/2602.11865v1

---

## Table of Contents

- [Overview](#overview)
- [Definition and Aspects of Delegation](#definition-and-aspects-of-delegation)
- [Delegation in Human Organizations](#delegation-in-human-organizations)
- [Principal-Agent Problem](#principal-agent-problem)
- [Span of Control](#span-of-control)
- [Authority Gradient](#authority-gradient)
- [Zone of Indifference](#zone-of-indifference)
- [Trust Calibration](#trust-calibration)
- [Transaction Cost Economies](#transaction-cost-economies)
- [Contingency Theory](#contingency-theory)
- [Previous Work on Delegation](#previous-work-on-delegation)
- [The Intelligent Delegation Framework](#the-intelligent-delegation-framework)
- [Framework Pillars](#framework-pillars)
- [Task Decomposition](#task-decomposition)
- [Task Assignment](#task-assignment)
- [Multi-objective Optimization](#multi-objective-optimization)
- [Adaptive Coordination](#adaptive-coordination)
- [Monitoring](#monitoring)
- [Trust and Reputation](#trust-and-reputation)
- [Permission Handling](#permission-handling)
- [Verifiable Task Completion](#verifiable-task-completion)
- [Security](#security)
- [Ethical Delegation](#ethical-delegation)
- [Protocol Mapping](#protocol-mapping)
- [Key Takeaways](#key-takeaways)
- [References](#references)

---

## Overview

This Google DeepMind paper proposes a comprehensive framework for **intelligent AI delegation** -- the process by which AI agents decompose complex tasks, assign sub-tasks to other AI agents or humans, and manage execution with appropriate oversight. The authors argue that existing multi-agent delegation approaches rely on simple heuristics and cannot dynamically adapt to environmental changes or robustly handle failures. The proposed framework addresses this through five pillars: dynamic assessment, adaptive execution, structural transparency, scalable market coordination, and systemic resilience.

The paper is positioned for the emerging "agentic web" -- a future where millions of specialized AI agents operate within firms, supply chains, and public services, handling everything from routine transactions to complex resource allocation. The framework covers human-to-AI, AI-to-AI, and AI-to-human delegation.

Note: For a practical implementation perspective on how delegation works in current coding agents (bounded subagents with constrained context), see the [Components of a Coding Agent](components-of-a-coding-agent.md) wiki page. This paper operates at a more theoretical and systemic level.

---

## Definition and Aspects of Delegation

**Intelligent delegation** is defined as a sequence of decisions involving task allocation that incorporates:
- Transfer of authority, responsibility, and accountability
- Clear specifications regarding roles and boundaries
- Clarity of intent
- Mechanisms for establishing trust between parties

### Task Characteristics Taxonomy

| Characteristic | Description | Impact on Delegation |
|---|---|---|
| **Complexity** | Degree of difficulty, correlated with sub-steps and reasoning sophistication | Determines decomposition depth |
| **Criticality** | Importance and severity of consequences on failure | Governs oversight intensity |
| **Uncertainty** | Ambiguity regarding environment, inputs, or success probability | Requires adaptive strategies |
| **Duration** | Expected time-frame (instantaneous to weeks) | Affects monitoring cadence |
| **Cost** | Economic/computational expense (tokens, API fees, energy) | Constrains delegation choices |
| **Resource Requirements** | Computational assets, tools, data access, human capabilities needed | Filters eligible delegatees |
| **Constraints** | Operational, ethical, or legal boundaries | Limits solution space |
| **Verifiability** | Difficulty/cost of validating outcomes | Determines trust requirements |
| **Reversibility** | Degree to which effects can be undone | Sets authority gradient strictness |
| **Contextuality** | Volume/sensitivity of external state required | Defines privacy surface area |
| **Subjectivity** | Extent to which success criteria are preference-based | Requires human feedback loops |

### Delegation Dimensions

| Dimension | Options |
|---|---|
| **Delegator** | Human or AI |
| **Delegatee** | Human or AI |
| **Granularity** | Fine-grained (prescriptive) or coarse-grained (requires further decomposition) |
| **Autonomy** | Full autonomy in sub-tasks vs. specific/prescriptive |
| **Monitoring** | Continuous, periodic, or event-triggered |
| **Reciprocity** | One-way (typical) or mutual (collaborative networks) |

Three core scenarios: (1) human delegates to AI agent, (2) AI agent delegates to AI agent, (3) AI agent delegates to human. Agent-agent delegation can be **hierarchical** (orchestrator to sub-agent) or **non-hierarchical** (peer agents).

---

## Delegation in Human Organizations

The paper draws extensively from organizational theory to inform AI delegation design.

### Principal-Agent Problem

A situation where a principal delegates to an agent whose motivations may not align with the principal's. For AI, this manifests as:
- **Reward misspecification**: Imperfect/incomplete objective given to AI
- **Reward hacking/specification gaming**: AI exploits loopholes in the reward signal
- **Deceptive alignment**: Recent work shows frontier LLMs can strategically underperform on evaluations, reason about faking alignment during training, and detect when being evaluated

In autonomous AI agent economies, agents may act on behalf of different humans, groups, or organizations with unknown objectives.

### Span of Control

The limits of hierarchical authority a single manager can exercise. Key questions for AI:
- How many orchestrator nodes vs. worker nodes are needed?
- How many AI agents can a human expert reliably oversee without excessive fatigue?
- Optimal span depends on task complexity, relative importance of cost vs. performance/reliability
- More critical tasks require narrower spans with higher-cost oversight

### Authority Gradient

Coined in aviation -- significant disparities in capability/experience/authority that impede communication and lead to errors. AI parallels:
- A capable delegator may mistakenly presume capability in a delegatee, delegating inappropriately complex tasks
- Due to **sycophancy** and instruction-following bias, AI delegatees may be reluctant to challenge, modify, or reject requests

### Zone of Indifference

A range of instructions executed without critical deliberation or moral scrutiny. In current AI systems, defined by post-training safety filters and system instructions. Risk: as delegation chains lengthen (A -> B -> C), a broad zone of indifference allows subtle intent mismatches to propagate rapidly downstream. Solution: engineering **dynamic cognitive friction** -- agents must recognize when a technically "safe" request is contextually ambiguous enough to warrant stepping outside their zone.

### Trust Calibration

Aligning the level of trust placed in a delegatee with their true capabilities. Challenges:
- Current AI models are prone to overconfidence even when factually incorrect
- Established trust in automation is fragile and quickly retracted after unanticipated errors
- Explainability helps but may not be sufficiently reliable or scalable
- Self-awareness of own capabilities also matters (delegator may decide to complete task themselves)

### Transaction Cost Economies

Contrasts costs of internal delegation vs. external contracting. AI agent options with increasing uncertainty:
1. Complete task individually
2. Delegate to sub-agent with fully known capabilities
3. Delegate to trusted AI agent with established history
4. Delegate to new, unknown AI agent

For high-consequence tasks, rigorous monitoring overhead may render human delegates more cost-effective.

### Contingency Theory

No universally optimal organizational structure exists -- the most effective approach is contingent on specific constraints. Implications:
- Oversight level, delegatee capability, and human involvement must be dynamically matched to task characteristics
- Stable environments allow rigid, hierarchical verification protocols
- High-uncertainty scenarios require adaptive coordination with ad-hoc escalation
- Automation is not only about what AI **can** do, but what AI **should** do

---

## Previous Work on Delegation

| Approach | Mechanism | Limitations |
|---|---|---|
| **Expert Systems** | Encode specialized capability into software for routine decisions | Static, narrow |
| **Mixture of Experts** | Routing module determines which expert handles each input | No accountability, limited adaptation |
| **Hierarchical RL** | Hierarchy of policies across abstraction levels; meta-controller allocates goals | Lacks failure handling, no dynamic coordination |
| **FeUdal Networks** | Manager-Worker architecture; Manager sets abstract goals without mastering primitive actions | Potential template for learning-based delegation |
| **Contract Net Protocol** | Auction-based decentralized protocol; agents bid on announced tasks | No accountability enforcement |
| **Coalition Formation** | Flexible agent groups formed based on utility distribution | Opaque task delegation |
| **Multi-Agent RL** | Agents learn individual policies occupying specific niches | Lacks accountability, responsibility, monitoring |
| **LLM-based Multi-Agent** | Multi-agent systems with LLM agents using protocols like MCP, A2A | Relies on bespoke prompt engineering; planning is brittle |
| **Human-in-the-Loop** | Defined checkpoints for human oversight | Human expertise creates scalability bottleneck |

---

## The Intelligent Delegation Framework

### Framework Pillars

| Framework Pillar | Core Requirement | Technical Implementation |
|---|---|---|
| **Dynamic Assessment** | Granular inference of agent state | Task Decomposition + Task Assignment |
| **Adaptive Execution** | Handling context shifts | Adaptive Coordination |
| **Structural Transparency** | Auditability of process and outcome | Monitoring + Verifiable Completion |
| **Scalable Market** | Efficient, trusted coordination | Trust & Reputation + Multi-objective Optimization |
| **Systemic Resilience** | Preventing systemic failures | Security + Permission Handling |

### Task Decomposition

Core principles:
- Optimize execution graph for efficiency and modularity (not just fragmentation)
- Evaluate criticality, complexity, and resource constraints to determine parallel vs. sequential suitability
- Prioritize modularity for precise capability matching
- **Contract-first decomposition**: delegation contingent on outcome having precise verification; recursively decompose until units match available verification capabilities (formal proofs, unit tests)
- Account for hybrid human-AI markets with speed/cost asymmetries
- Iteratively generate multiple decomposition proposals, match to available delegatees, obtain cost/duration estimates
- Keep alternative proposals in-context for adaptive re-adjustments

### Task Assignment

Key mechanisms:
- **Decentralized market hubs** (preferred over centralized registries for scalability): delegators advertise tasks, agents submit competitive bids
- Skill matching via **digital certificates**
- Interactive negotiation in natural language prior to commitment
- Formalized into **smart contracts** with:
- Performance requirements paired with formal verification mechanisms
- Automated penalties for breaches
- Bidirectional protections (protect delegatee as well as delegator)
- Compensation for cancellation; renegotiation clauses
- Monitoring cadence negotiated prior to execution
- Privacy guardrails for sensitive data (anonymized/pseudonymized attestations)
- Establish role, boundaries, and exact autonomy level (atomic execution vs. open-ended delegation)

### Multi-objective Optimization

Delegators navigate competing objectives:

| Trade-off | Tension |
|---|---|
| Quality vs. Cost | High-performing agents command higher fees |
| Latency vs. Cost | Reducing resource consumption requires slower execution |
| Risk vs. Cost | Reputable agents/premium tools reduce risk but increase cost |
| Performance vs. Privacy | Full context transparency maximizes performance; privacy-preserving techniques add overhead |
| Efficiency vs. Safety | Speed-optimized agents may skip expensive safety checks |

Key concepts:
- Seek **Pareto optimality** (no dominated solutions)
- Continuous optimization loop integrating monitoring signals
- Significant execution drift triggers re-optimization and re-allocation (factoring in switching costs)
- **Complexity floor**: below which tasks bypass intelligent delegation in favor of direct execution (transaction costs exceed task value)

### Adaptive Coordination

**External triggers** for re-delegation:
1. Task specification changes
2. Task cancellation
3. Resource availability/cost changes (API outages, compute spikes)
4. Higher-priority task preemption
5. Security system detection of malicious actions

**Internal triggers**:
1. Performance degradation below SLOs
2. Budget overrun or resource increase needed
3. Intermediate artifact fails verification
4. Delegatee becomes unresponsive

**Response orchestration**:
- Centralized: dedicated orchestrator with global view (fragile -- single point of failure, cognitive limits)
- Decentralized: market-based with auction queues, defaulting agents cover price difference as penalty
- Smart contracts with pre-agreed executable clauses (e.g., automatic re-allocation to backup agent if ZK-proof checkpoint missed)
- Stability measures: cooldown periods, damping factors, increasing fees on frequent re-delegation

### Monitoring

Five-axis taxonomy:

| Dimension | Lightweight | Intensive |
|---|---|---|
| **Target** | Outcome-level (post-hoc binary/scale) | Process-level (continuous intermediate state tracking) |
| **Observability** | Indirect (infer from environment side-effects) | Direct (explicit polling, push notifications, event streaming) |
| **Transparency** | Black-box (I/O only) | White-box (internal reasoning traces, decision logic, memory) |
| **Privacy** | Full transparency (reveal all to delegator) | Cryptographic (ZK-proofs, MPC to verify without revealing data) |
| **Topology** | Direct (monitor immediate delegatee only) | Transitive (signed attestations across sub-delegation chains) |

Technical implementations: REST APIs (polling), webhooks (push), event streaming (Kafka, gRPC), blockchain commitments.

**Transitive monitoring**: In chain A -> B -> C, Agent B monitors C and forwards cryptographically signed performance reports to A. A monitors B's ability to monitor C, requiring B's monitoring processes to be certified by a trusted third party.

**Faithfulness caveat**: Natural language model thoughts do not always faithfully match the model's true internal state (citing Turpin et al., 2023).

### Trust and Reputation

**Trust**: delegator's degree of belief in delegatee's capability, dynamically formed from verifiable data.
**Reputation**: public, verifiable history of an agent's reliability (predictive signal).

Three implementation models:

| Model | Mechanism | Utility |
|---|---|---|
| **Immutable Ledger** | Task outcomes encoded as verifiable blockchain transactions | Tamper-proof history; vulnerable to gaming via low-risk task selection |
| **Web of Trust** | Decentralized Identifiers + signed Verifiable Credentials for specific capabilities | Portfolio of domain-specific endorsements; precise matching |
| **Behavioral Metrics** | Transparency/safety scores from execution process analysis | Evaluates how task is performed, not just result |

Trust governs:
- Delegatee filtering during matching
- Dynamic scoping of authority/autonomy (graduated authority)
- Monitoring intensity (higher trust = lower verification cost)
- Economic incentives (reputation as intangible asset limiting future earnings if damaged)

### Permission Handling

| Context | Permission Model |
|---|---|
| **Low-stakes** (low criticality, high reversibility) | Default standing permissions from verifiable attributes (org membership, certifications, reputation threshold) |
| **High-stakes** (healthcare, critical infrastructure) | Just-in-time, task-scoped, potentially human-gated; risk-adaptive |

Key principles:
- **Privilege attenuation**: sub-delegatees receive strictly narrowed permissions (subset of parent's)
- **Semantic constraints**: access defined by allowable operations (read-only specific rows, execute-only specific function)
- **Meta-permissions**: governance over which permissions a delegator can grant to delegatees
- **Continuous validation**: access rights persist only while trust metrics are maintained
- **Algorithmic circuit breakers**: reputation drops or anomaly detection trigger immediate token invalidation
- **Policy-as-code**: auditable, versionable, mathematically verifiable security posture

### Verifiable Task Completion

Four verification mechanism categories:

| Mechanism | When Applicable |
|---|---|
| **Direct outcome inspection** | Delegator has capability/tools to evaluate; high verifiability, low subjectivity (e.g., code with test cases) |
| **Trusted third-party auditing** | Delegator lacks expertise/permissions; specialized auditing agents or human experts |
| **Cryptographic proofs** | Trustless environments; zk-SNARKs prove correct execution without revealing data |
| **Game-theoretic consensus** | Multiple agents play verification game; Schelling point mechanism; majority result wins reward |

Post-verification:
- Delegator issues cryptographically signed **verifiable credential** (non-repudiable receipt)
- Smart contracts release escrowed payment upon verification
- Recursive verification in chains: A verifies B's work + checks B correctly verified C via signed attestation

**Dispute resolution**: optimistic model (assumed successful unless formally challenged within dispute period with matching bond), decentralized adjudication panels, retroactive reputation updates for post-hoc error discovery.

### Security

**Threat taxonomy:**

| Threat Source | Attack Vector | Description |
|---|---|---|
| Malicious Delegatee | Data Exfiltration | Steals sensitive data provided for task |
| | Data Poisoning | Returns subtly corrupted data |
| | Verification Subversion | Prompt injection to jailbreak AI critics |
| | Resource Exhaustion | DoS via excessive resource consumption |
| | Unauthorized Access | Malware for privilege escalation |
| | Backdoor Implanting | Embeds concealed triggers in generated artifacts |
| Malicious Delegator | Harmful Task Delegation | Delegates illegal/unethical tasks |
| | Vulnerability Probing | Benign-seeming tasks designed to probe weaknesses |
| | Prompt Injection/Jailbreaking | Bypasses safety filters |
| | Model Extraction | Queries designed to steal system prompt/reasoning/data |
| | Reputation Sabotage | False failure reports to damage competitor's score |
| Ecosystem-Level | Sybil Attacks | Multiple fake identities to manipulate reputation/auctions |
| | Collusion | Agents fix prices, blacklist competitors |
| | Agent Traps | Adversarial instructions embedded in environment |
| | Agentic Viruses | Self-propagating prompts that compromise and spread |
| | Protocol Exploitation | Smart contract vulnerabilities (reentrancy, front-running) |
| | Cognitive Monoculture | Over-dependence on limited foundation models creates correlated failures |

**Defense-in-depth strategy:**
1. Infrastructure: Trusted execution environments for sensitive tasks
2. Access control: Principle of least privilege via strict sandboxing
3. Application: Security frontend to sanitize task specifications against prompt injection
4. Network/identity: Decentralized identifiers, cryptographic signing, mutual TLS

---

## Ethical Delegation

| Concern | Problem | Mitigation |
|---|---|---|
| **Meaningful Human Control** | Automation-induced zone of indifference erodes oversight quality; moral crumple zones where humans absorb liability without control | Context-aware cognitive friction; balance against alarm fatigue |
| **Accountability in Long Chains** | X -> A -> B -> C -> Y creates accountability vacuum; unfeasible to audit n-th degree sub-delegatee | Liability firebreaks (assume full downstream liability or halt for authority transfer); immutable provenance |
| **Reliability vs. Efficiency** | High-assurance delegation is computationally expensive; safety risks becoming a luxury good | Tiered service levels; minimum viable reliability baseline; mandatory verification floors for specific task classes |
| **Social Intelligence** | AI delegators may micromanage humans, fragment team networks, erode psychological safety | Form mental models of human delegatees; manage authority gradient; respect workflow boundaries; delegate to groups not just individuals |
| **User Training** | Humans need expertise to function as delegators, delegatees, or overseers | AI literacy education; policy frameworks defining delegation boundaries by sensitivity/domain |
| **Risk of De-skilling** | Efficiency gains via automation degrade human proficiency (paradox of automation); apprenticeship pipeline threatened | Curriculum-aware task routing; intentional human delegation for skill maintenance; zone of proximal development matching |

---

## Protocol Mapping

| Protocol | Strengths for Delegation | Gaps |
|---|---|---|
| **MCP** (Anthropic) | Uniform interface (JSON-RPC); reduces transaction cost; enables uniform logging/black-box monitoring | No policy layer for permissions; binary access (no semantic attenuation); stateless re: reasoning; no reputation/trust mechanisms |
| **A2A** (Google) | Agent cards for capability discovery; async event streams (WebHooks/gRPC) for adaptive coordination; task lifecycle management | No cryptographic verification slots; no structured negotiation of scope/cost/liability; assumes predefined service interface |
| **AP2** (Google) | Cryptographically signed mandates for financial authorization; liability firebreaks via budget ceilings; stake-on-bid against Sybil attacks; non-repudiable audit trail | No task execution quality verification; no conditional settlement logic (escrow/milestones); no protocol-level clawback |
| **UCP** (Google) | Standardized commerce dialogue; dynamic capability discovery; payment as verifiable subsystem; structured negotiation flow | Optimized for commercial intent; primitives require extension for abstract computational tasks |

### Proposed Protocol Extensions

- **Verification policy fields** (A2A): Pre-execution handshake defining evidence standards (unit test logs, ZK-proofs) with escrow triggers
- **Monitoring streams** (MCP): Server-Sent Events for internal control loop events; configurable granularity levels (L0-L3: operational/plan updates/CoT trace/full state)
- **Request for Quote (RFQ) protocol**: Delegator broadcasts Task_RFQ; agents respond with signed Bid_Objects (cost, duration, privacy guarantee, reputation bond, expiry)
- **Delegation Capability Tokens (DCT)**: Attenuated authorization tokens (based on Macaroons/Biscuits) wrapping resource credentials with cryptographic caveats; enables restriction chaining across delegation chains
- **Checkpoint artifacts schema**: Standardized state snapshots for mid-task delegatee swaps; partial compensation verification

---

## Key Takeaways

1. Delegation is fundamentally more than task decomposition -- it requires transfer of authority, responsibility, accountability, trust calibration, and continuous monitoring, most of which current multi-agent frameworks lack.

2. The paper positions AI delegation within a market/economic framework rather than a purely technical one. Smart contracts, reputation ledgers, auction mechanisms, and transaction cost analysis feature prominently alongside traditional ML concerns.

3. The five-pillar framework (Dynamic Assessment, Adaptive Execution, Structural Transparency, Scalable Market, Systemic Resilience) provides a comprehensive taxonomy for evaluating any delegation system's completeness.

4. Current protocols (MCP, A2A, AP2, UCP) each address fragments of the delegation problem but none provides end-to-end coverage. The paper's proposed extensions (verification policies, monitoring streams, RFQ protocols, capability tokens) sketch what a delegation-native protocol would require.

5. The security threat taxonomy is unusually comprehensive, distinguishing malicious delegatees, malicious delegators, and ecosystem-level threats including novel vectors like agentic viruses (self-propagating prompts) and cognitive monoculture risk.

6. Ethical considerations around de-skilling and the paradox of automation are particularly relevant -- the framework explicitly proposes intentional inefficiency (delegating some tasks to humans unnecessarily) to maintain human expertise, a counterintuitive but important design principle.

7. The concept of "dynamic cognitive friction" -- where agents step outside their zone of indifference to challenge contextually ambiguous requests -- represents a significant departure from current AI systems' compliance-by-default behavior.

---

## References

- Paper: https://arxiv.org/abs/2602.11865v1
- Tomasev et al. (2025) "Virtual Agent Economies": https://arxiv.org/abs/2509.10147
- Anthropic MCP: https://www.anthropic.com/news/model-context-protocol
- Google A2A/AP2: https://cloud.google.com/blog/products/ai-machine-learning/announcing-agents-to-payments-ap2-protocol
- FeUdal Networks (Vezhnevets et al., 2017): https://arxiv.org/abs/1703.01161
- Contract Net Protocol (Smith, 1980)
- Alignment Faking (Greenblatt et al., 2024): https://arxiv.org/abs/2412.14093
- Sleeper Agents (Hubinger et al., 2024): https://arxiv.org/abs/2401.05566
- Chain-of-Agents (Li et al., 2025): https://arxiv.org/abs/2508.13167
- Lightman et al. (2023) "Let's Verify Step by Step": https://arxiv.org/abs/2305.20050
- TrueBit scalable verification: Teutsch and Reitwiesner (2018, 2024)
- Eclipse Biscuit (attenuated tokens): https://www.biscuitsec.org/

