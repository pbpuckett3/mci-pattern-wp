# Model Context Intelligence
## Intelligent Orchestration for the Multi-Model Era

**Paul Puckett • December 2025**

---

## Executive Summary

The AI industry has built itself on a concentrated foundation. Frontier model inference depends on high-end GPUs manufactured by a single company, fabricated by a single foundry, packaged with capacity already sold out through 2026. The April 2024 Taiwan earthquake caused only brief production pauses (TSMC's engineering held), but the deeper vulnerability is geopolitical: Taiwan Strait tensions threaten access to infrastructure that cannot be quickly replicated elsewhere.

Meanwhile, organizations are routing every task, from calendar lookups to code generation to threat detection, through these same constrained frontier models. Stanford's 2025 AI Index reports that inference costs for GPT-3.5-equivalent performance dropped 280-fold between 2022 and 2024. Yet enterprise AI budgets continue rising as usage growth outpaces efficiency gains. The paradox: per-token costs are falling while total spend accelerates. Cheaper models, used for everything, cost more than right-sized models used appropriately.

This is not sustainable. And it is not necessary.

Specialized models under 3 billion parameters now match frontier performance on bounded tasks, at a fraction of the cost, latency, and infrastructure dependency. The question is no longer whether multi-model architectures work. The question is how to orchestrate them: when to decompose tasks, how to route subtasks, what to optimize for, and how to learn across security and compliance boundaries.

This paper introduces **Model Context Intelligence (MCI)**: an architectural pattern for multi-model orchestration. MCI synthesizes ideas from agent frameworks, model routing research, and enterprise patterns into opinionated positions on decomposition (SCALE rubric), routing (compliance gate then CLASSic optimization), context management, and cross-boundary learning. It complements existing SDKs like Microsoft Agent Framework, LangChain, and CrewAI, providing the decision logic they leave open.

---

## 1. The Fragility Problem

The current AI architecture concentrates risk at every layer.

**Infrastructure concentration.** Nvidia controls approximately 80% of the AI accelerator market. TSMC fabricates these chips. Advanced packaging capacity (CoWoS) is fully booked through 2025 and constrained through 2026. The April 2024 Taiwan earthquake caused only brief production pauses (TSMC's earthquake-resistant designs held), but it highlighted the concentration risk. The more pressing concern is geopolitical: Taiwan Strait tensions could disrupt access to capacity that cannot be quickly replicated elsewhere.

**Operational concentration.** Most production applications route every request through frontier models requiring this concentrated infrastructure. When your calendar assistant, your code reviewer, your fraud detector, and your research agent all depend on the same GPU clusters, you've created correlated failure modes across your entire AI portfolio.

**Economic unsustainability.** Enterprise AI spending continues accelerating even as per-token costs fall; usage growth outpaces efficiency gains (Stanford AI Index 2025, Figure 4.3.12). Organizations are paying frontier-model prices for tasks that specialized models handle at 10-50x lower cost (observed 2025 cloud pricing for sub-3B vs. 70B+ parameter models). The economics work only if you assume infinite infrastructure availability at current prices. That assumption is already breaking.

The industry recognizes this. McKinsey projects $6.7 trillion in data center infrastructure investment needed by 2030 to meet compute demand, with AI workloads driving the majority of growth. But building infrastructure takes years. The gap between demand and capacity is structural, not temporary.

The answer is not more infrastructure for the same architecture. The answer is a different architecture that uses infrastructure more intelligently.

**Specialized models are ready.** Phi-3-mini (3.8B parameters) matches Mixtral 8x7B on many benchmarks. Fine-tuned Llama variants approach GPT-4 on domain tasks like text-to-SQL. MobileLLM enables sub-200ms inference on edge devices. The capability exists. What's missing is the intelligence to use it.

Orchestration is not new. Load balancers route traffic. API gateways manage requests. Workflow engines sequence tasks. But these are mechanical orchestrators: they route based on static rules, round-robin algorithms, or simple capacity metrics. They don't understand what they're routing.

Intelligent orchestration is different. It understands task semantics well enough to decompose complex requests into right-sized subtasks. It knows which sub-agents are compliant for which data types before considering performance. It learns from outcomes, not just "did it succeed?" but "how did cost, latency, and accuracy compare to alternatives?" It adapts routing based on observed performance, not just configured rules. It maintains context across multi-step workflows so downstream tasks benefit from upstream decisions.

The gap in the market is not orchestration. It's the decision intelligence that makes orchestration effective: systematic rubrics for decomposition, compliance-aware routing, multi-dimensional optimization, and learning that works across trust boundaries.

That decision intelligence is what MCI provides.

---

## 2. The Power of Specialization

Specialized models have moved from "good enough" to "strictly superior" on bounded tasks. This shift stems from a fundamental insight: a model trained or fine-tuned on domain-specific data, with an architecture optimized for specific task patterns, can outperform a general-purpose model orders of magnitude larger.

Why does this work?

**Focused training data.** A 3B parameter model fine-tuned on millions of medical case studies develops deeper pattern recognition for clinical reasoning than a 500B model trained on general web text. The smaller model has seen more relevant examples per parameter.

**Optimized architectures.** Models designed for specific tasks can make architectural trade-offs that general-purpose models cannot. Code models can optimize for syntax patterns. Classification models can optimize for decision boundaries. Retrieval models can optimize for embedding quality.

**Efficient inference.** Smaller models run faster, cheaper, and on more hardware. A model that's 10% more accurate but 50x more expensive isn't better-it's just more expensive. For bounded tasks where the specialized model matches or exceeds accuracy, there's no trade-off at all.

Recent benchmarks illustrate the trend. While specific results vary by evaluation methodology, the pattern is consistent:

| Model | Size | Domain / Task | Observation |
|-------|------|---------------|-------------|
| Microsoft Phi-3-mini | 3.8B | General reasoning | Competitive with Mixtral 8x7B and GPT-3.5 on many benchmarks |
| Fine-tuned Llama variants | 7-8B | Text-to-SQL (Spider) | Approaching or matching GPT-4 accuracy with domain tuning |
| Granite-Code models | 3-8B | Code generation | Strong HumanEval performance relative to size |
| MobileLLM | 125M-350M | On-device classification | Enables sub-200ms inference on mobile hardware |
| Domain fine-tuned SLMs | 1-8B | Legal, medical, financial | Often outperform base models 10-40x larger on domain tasks |

The implication: the optimal architecture is not one model that does everything. It is many models, each excelling at what it does best, coordinated by an intelligent orchestration layer.

This is not a new insight. Frameworks like LangChain, AutoGen, CrewAI, and Microsoft Agent Framework have been exploring multi-model coordination since 2023. What MCI contributes is an opinionated synthesis: specific positions on decomposition, routing, context management, and learning that are rarely given concrete rubrics in existing frameworks.

---

## 3. Introducing Model Context Intelligence

Model Context Intelligence is an architectural pattern for coordinating specialized sub-agents across heterogeneous model tiers, not a single product or implementation.

### What MCI Is (and Isn't)

MCI is not a claim of invention. The component ideas exist across multiple frameworks and research efforts: LangChain pioneered tool chaining and agent patterns; AutoGen introduced multi-agent coordination with learning; CrewAI developed task decomposition and crew orchestration; vLLM's Semantic Router (now Signal-Decision) demonstrated learned model routing; Ray Serve and Kubernetes established patterns for distributed inference serving.

What MCI contributes is synthesis and opinion. We take specific positions on how these capabilities should combine for enterprise and mission-critical deployments: which rubrics for decomposition (SCALE) and routing (CLASSic), how context should persist, how learning should work across security boundaries, what the component boundaries should be. These are opinionated choices, informed by regulated-industry requirements, that are less explicitly codified in existing frameworks.

### Why "Model Context Intelligence"?

The naming is deliberately parallel to **Model Context Protocol (MCP)**. Where MCP solves tool and data integration (giving models access to external capabilities), MCI solves orchestration intelligence: deciding which sub-agents handle which tasks, maintaining context across workflows, and learning from outcomes.

The term captures three essential dimensions:

**Model**: The architecture orchestrates across multiple models of different sizes, architectures, and specializations, from micro language models at the edge to frontier models in secure clouds.

**Context**: Intelligence is context-aware across multiple dimensions: conversation history, workflow state, user preferences, security constraints, resource budgets, and observed performance patterns.

**Intelligence**: The architecture exhibits adaptive intelligence through recursive learning. It observes its own execution, learns from outcomes, and continuously optimizes decomposition strategies, routing decisions, failure recovery policies, and synthesis approaches.

MCP and MCI are complementary layers in a complete AI architecture. MCP provides the foundation for tool access; MCI provides the intelligence for orchestration.

---

## 4. Five-Layer Architecture

Before diving into components, here's how MCI fits into the overall stack:

**Layer 5 - Application Interface:** REST/GraphQL/Streaming APIs. Abstracts all orchestration complexity from applications.

**Layer 4 - MCI Orchestration:** Eight core components that decompose tasks, route to sub-agents, manage context, coordinate execution, and learn from outcomes. This is where intelligence lives.

**Layer 3 - Sub-Agent Layer:** Specialized sub-agents implementing a standard SDK interface. Examples: context retrieval, calendar analysis, threat detection, code generation. Each sub-agent wraps one or more models with domain-specific logic.

**Layer 2 - Heterogeneous Model Tier:** Micro models (thousands to tens of millions of parameters, for edge and embedded), small models (hundreds of millions to a few billion, for local inference), and frontier models (tens of billions and up, for complex reasoning). Accessed via sub-agents, not directly.

**Layer 1 - MCP Foundation:** Tool and data integration layer. Sub-agents access external tools and data sources via MCP through the SDK.

### Sub-Agents and Models are Decoupled

A critical architectural distinction: sub-agents and models are not the same thing. Sub-agents are deployment units that wrap models. The same model can power multiple sub-agents with entirely different characteristics.

Consider a Llama 70B model deployed four ways:

| Sub-Agent | Model | Deployment | Compliance |
|-----------|-------|------------|------------|
| Legal-Analysis-Gov | Llama 70B | Azure Government | FedRAMP High, IL4 |
| Legal-Analysis-Commercial | Llama 70B | AWS US-East | SOC2 |
| Legal-Analysis-EU | Llama 70B | AWS Frankfurt | GDPR |
| Legal-Analysis-Classified | Llama 70B | On-prem enclave | IL5 |

Same model. Same fine-tuning. Four different sub-agents with different compliance postures, data residency, and availability characteristics.

This decoupling matters because MCI routes to sub-agents, not models. The routing decision considers both deployment properties (compliance, residency, availability) and observed performance (cost, latency, accuracy). A sub-agent's performance reflects the combination of its underlying model and its deployment environment.

**Sub-agents encapsulate capability, not just access.** The wrapper ecosystem’s frustration is that you can connect to dozens of models but lose critical capabilities when you leave the major providers. Grounding (RAG), tool use, structured outputs, function calling: these aren’t standardized across models. A sub-agent solves this by encapsulating the model plus its capability bindings. The RAG pipeline, the tool definitions, the output schemas, the prompt templates that make a particular model work well for a particular task: all of that lives in the sub-agent. The orchestration layer routes to a sub-agent that can do “contract analysis with clause extraction.” It doesn’t know or care whether that sub-agent uses RAG, fine-tuning, or a 200-line system prompt to get there. This isolation means you can swap implementations, upgrade models, or change grounding strategies without touching orchestration logic.

---

## 5. The Eight Components of MCI

MCI comprises eight core components. Each has a specific role in the orchestration flow.

### Component 1: Workflow Orchestrator (Go)

The Workflow Orchestrator is the entry point for every request. Its job: decide whether to decompose a task into subtasks, and if so, how.

**Language choice:** Go. Workflow orchestration engines like Temporal and Cadence are built in Go. The language offers strong concurrency, simple deployment, and adequate performance for decision logic.

**Alternatives to consider:** Python if the orchestrator relies heavily on LLM-based classification for SCALE assessment, since Python offers tighter integration with ML tooling. Java/Kotlin for enterprises with existing JVM infrastructure and teams.

This is not a trivial decision. Decomposition adds orchestration overhead. For simple tasks, sending directly to a capable model may be faster and cheaper. For complex tasks, decomposition enables parallel execution and right-sized model selection. For safety-critical tasks, decomposition may be too risky-or may require redundant validation paths.

#### The SCALE Rubric

To make this decision systematically, the Workflow Orchestrator applies the **SCALE** rubric:

| Dimension | Question |
|-----------|----------|
| **Structure** | Is this task naturally decomposable into meaningful subtasks? |
| **Consequence** | What happens if we fail? |
| **Accuracy** | What error tolerance exists? |
| **Latency** | What are the time constraints? |
| **Experience** | How proven is this pattern? (Volume + Confidence) |

#### How SCALE Gets Applied

The Orchestrator assesses each dimension through a combination of classification, policy lookup, and historical data:

**Step 1: Assess each dimension**

| Dimension | Assessment Method | Output |
|-----------|-------------------|--------|
| Structure | LLM classifier or pattern match against known task types | Atomic / Decomposable / Unknown |
| Consequence | Policy config + task metadata (e.g., domain tags, user role) | Advisory / Operational / Safety-Critical / Life-Safety |
| Accuracy | Derived from task type or explicit requirement in request | Approximate OK / Must Be Correct / Zero Defect |
| Latency | Explicit SLA in request or inferred from context | Real-time (<500ms) / Interactive (<5s) / Batch (>5s) |
| Experience | Lookup against decomposition template history | High (>1000 executions, >95% success) / Medium / Low / Novel |

**Step 2: Apply decision logic**

The Orchestrator applies these assessments through a decision hierarchy where Consequence dominates.

For life-safety tasks, the Orchestrator only decomposes if there's a high-experience template with redundant validation built in. Otherwise, it routes the entire task to the highest-accuracy model available because decomposition risk is too high.

For atomic tasks (those that can't meaningfully be broken apart), the Orchestrator skips decomposition entirely.

For tasks with low experience and operational or higher consequence, the Orchestrator avoids decomposition due to insufficient confidence in the pattern.

For decomposable tasks with adequate experience, the Orchestrator applies the best-match template and parallelizes subtasks where latency requirements demand it and dependencies allow.

For unknown task structures, the Orchestrator routes to a frontier model for single-pass handling and logs the interaction for pattern discovery.

**Step 3: Build execution plan**

If decomposing, the Orchestrator:
- Selects decomposition template (or synthesizes from similar patterns)
- Identifies subtask dependencies (A must complete before B)
- Determines parallelization opportunities
- Passes plan to Sub-Agent Coordinator

#### Key Insight: Consequence as Override

When tasks have safety-of-navigation or loss-of-life implications, Consequence dominates all other factors. Accuracy becomes paramount. Cost becomes almost irrelevant. The Orchestrator may bypass decomposition entirely or mandate redundant validation paths regardless of efficiency.

This is why SCALE treats Consequence as a governing constraint, not just another weighted factor.

---

### Component 2: Intelligent Router (Rust)

Once the Orchestrator has decided to decompose (or not), each task or subtask needs to be routed to a sub-agent. The Intelligent Router makes this decision.

**Language choice:** Rust. The Router sits on the critical path of every request. Industry precedent from vLLM Semantic Router and Envoy demonstrates that latency-critical routing benefits from Rust's performance characteristics and memory safety guarantees.

**Alternatives to consider:** Go is viable if latency requirements are relaxed (interactive rather than real-time). We do not recommend higher-level languages for this component; the Router's position on the critical path makes performance non-negotiable.

#### Routing to Sub-Agents, Not Models

Because sub-agents are deployment units that wrap models, routing evaluates the whole sub-agent, not the underlying model in isolation. A sub-agent's performance reflects the combination of its model capability and its deployment context.

The Router applies a two-phase selection process: compliance gating, then performance optimization.

#### Phase 1: Compliance Gate (Pass/Fail)

Compliance is a hard gate. Before any optimization, the Router filters to eligible sub-agents only.

Compliance encompasses all regulatory and policy requirements: Does the sub-agent meet required regulatory standards (HIPAA, PCI-DSS, FedRAMP, SOC2)? Can it handle the data's classification or sensitivity level? Does data stay within required jurisdictions? Does it have required certifications?

Any sub-agent that fails compliance requirements for the task is excluded. Only compliant sub-agents proceed to Phase 2.

#### Phase 2: CLASSic Performance Scoring

For compliant sub-agents, the Router applies **CLASSic** (Aisera, ICLR 2025) to evaluate observed performance:

| Dimension | What It Measures |
|-----------|------------------|
| **Cost** | Operational expenses: API usage, token consumption, infrastructure |
| **Latency** | End-to-end response times |
| **Accuracy** | Correctness for this task type |
| **Stability** | Consistency across diverse inputs and conditions |
| **Security** | Resilience against adversarial inputs, prompt injections, data leaks |

These dimensions are measured at the sub-agent level as observed in production, not abstract model benchmarks. A sub-agent's CLASSic scores reflect the combination of its underlying model and its deployment environment.

#### Model Benchmarks vs. CLASSic: Design-Time vs. Runtime

Two levels of evaluation matter here:

**Model benchmarks** (MMLU, HumanEval, MATH, GPQA, SWE-Bench) are agent-agnostic tests of raw model capability. They answer questions like: Is this model good at code generation? How well does it reason about math? These benchmarks inform **design-time decisions**, specifically which model to wrap when building a sub-agent for a particular task type. A code-review sub-agent might wrap a model with strong HumanEval scores; a knowledge-QA sub-agent might prioritize MMLU performance.

**CLASSic** evaluates sub-agent performance across operational dimensions. It answers questions like: How does this sub-agent actually perform in production on cost, latency, accuracy, stability, and security? CLASSic informs **runtime routing decisions**, specifically which sub-agent to select for a given task.

Model benchmarks are inputs to sub-agent design. CLASSic is the basis for sub-agent selection. The Router operates on CLASSic scores observed in production, not raw model benchmarks.

| Dimension | Scoring Method | Score Range |
|-----------|----------------|-------------|
| Cost | Observed cost per task type | $ (lower is better) |
| Latency | Historical p95 latency for similar task types | ms (lower is better) |
| Accuracy | Historical success rate for this task type | 0.0 - 1.0 (higher is better) |
| Stability | Variance in accuracy over trailing window | σ (lower is better) |
| Security | Historical resistance to adversarial probes | 0.0 - 1.0 (higher is better) |

#### Phase 3: Apply Workflow Weights

Different workflows prioritize differently. The Router applies weights from the workflow configuration, computing a weighted score across all five dimensions. The sub-agent with the highest combined score is selected.

Example weight profiles:

| Workflow Type | Cost | Latency | Accuracy | Stability | Security |
|---------------|------|---------|----------|-----------|----------|
| Real-time chat | 0.1 | 0.4 | 0.25 | 0.1 | 0.15 |
| Financial analysis | 0.1 | 0.1 | 0.4 | 0.2 | 0.2 |
| Bulk processing | 0.4 | 0.1 | 0.3 | 0.1 | 0.1 |
| Safety-critical | 0.0 | 0.1 | 0.5 | 0.2 | 0.2 |

#### Hybrid Routing: Compliance as Gate, Performance as Gradient

The Router implements a hybrid approach:
- **Compliance gate** enforces regulatory and policy requirements, auditable and explainable to regulators
- **CLASSic optimization** improves performance based on observed outcomes, providing adaptive efficiency within compliant boundaries

Compliance is a gate. Performance is a gradient.

---

### Component 3: Context Manager (Rust)

The Context Manager maintains all state across the workflow. Sub-agents are stateless by design-they receive context, do work, return results. The Context Manager owns the memory.

**Language choice:** Rust. Context is critical state. Corruption or race conditions in context management can cause cascading failures across the entire workflow. Rust's ownership model and compile-time guarantees provide correctness assurances that matter for this component.

**Alternatives to consider:** Go with careful concurrency design could work, but requires discipline that Rust enforces at compile time. If the Context Manager is primarily an application layer over a proven store like Redis or etcd, the language matters less because the storage layer handles correctness. We do not recommend dynamically-typed languages for this component.

#### What It Tracks

| Context Type | Examples | Persistence |
|--------------|----------|-------------|
| Conversation history | Full message log across turns | Indefinite (default) |
| Workflow state | Subtask completion, intermediate results, dependencies | Request-scoped |
| User preferences | Writing style, risk tolerance, formatting | Cross-session |
| Security constraints | Classification level, allowed operations | Policy-defined |
| Resource budgets | Token limits, cost ceiling, latency SLA | Request-scoped |

#### Architectural Position: MCI Owns Memory

**Default to indefinite persistence.** Too many cycles with models are spent relearning things already known. If MCI is the orchestration layer, then MCI is where context lives-not the models. The models are stateless workers.

Regulated environments can configure retention policies (flush after request, TTL, scope limits) as overrides. But the architectural default assumes persistence because relearning is wasteful.

#### Representation: Declarative Structure, Learned Content

**Structure is declarative.** The categories of context (user preferences, workflow state, conversation history, constraints) are schema-defined-inspectable, queryable, auditable.

**Content can be learned.** Within those categories, content may use learned representations: summaries for long histories, embeddings for semantic retrieval, compression for efficiency.

You always know what kinds of context exist. How that context is internally represented can vary.

---

### Component 4: Sub-Agent Coordinator (Go)

The Coordinator executes the plan created by the Orchestrator, invoking sub-agents in the correct order and handling failures.

**Language choice:** Go. Goroutines and channels are ideal for managing concurrent sub-agent invocations with clean cancellation and timeout handling. Go's concurrency model maps naturally to parallel and conditional execution patterns.

**Alternatives to consider:** Elixir/Erlang (OTP) if fault tolerance is the primary design constraint, since the "let it crash" supervision model excels at failure recovery. Java with virtual threads (Project Loom) for enterprises with JVM infrastructure. Rust with Tokio for deployments already using Rust elsewhere.

#### Execution Modes

| Mode | When Used | Behavior |
|------|-----------|----------|
| Sequential | Subtask B depends on output of A | Execute A, wait, execute B |
| Parallel | Subtasks A and B are independent | Execute simultaneously, reduce latency |
| Conditional | Subtask B only needed if A returns certain result | Evaluate A output, decide on B |

#### Failure Handling

The Coordinator implements adaptive failure handling. When a sub-agent fails, the Coordinator first attempts retries if any remain, since the failure may be transient. If retries are exhausted, it checks whether an alternative sub-agent can handle the task and routes accordingly. If no alternative exists but the workflow can tolerate partial results, the Coordinator continues with a degraded response. If human escalation is configured, it queues for review. Only when all recovery options are exhausted does the workflow fail with a clear error.

Failure strategies are learned over time. The Adaptive Learning System observes which recovery approaches work for which failure types and updates Coordinator policies.

---

### Component 5: Result Synthesizer (Go)

When a workflow involves multiple sub-agents, their outputs must be combined into a coherent response. The Synthesizer handles this.

**Language choice:** Go. Rule-based synthesis is straightforward business logic that doesn't require specialized language features. Using the same language as the Coordinator simplifies deployment and team expertise.

**Alternatives to consider:** Python if synthesis requires ML-assisted merging or semantic understanding beyond rule-based policies. In that case, the Synthesizer may call ML models for intelligent merging rather than implementing it directly.

#### Synthesis Challenges

- Different sub-agents may use different formats or terminology
- Outputs may partially overlap or conflict
- Confidence levels vary across sub-agents
- User expects unified response, not a list of fragments

#### Conflict Resolution

Conflicts are resolved via pre-configured MCI policies:

| Conflict Type | Resolution Strategy |
|---------------|---------------------|
| Factual disagreement | Prefer higher-confidence sub-agent; flag uncertainty |
| Format mismatch | Normalize to workflow-specified format |
| Partial overlap | Deduplicate, merge unique information |
| Missing subtask output | Note gap if critical, omit if optional |

The Synthesizer does not make judgment calls-it follows policy. This keeps synthesis auditable and predictable.

---

### Component 6: Response Validator (Go)

Before returning results, the Validator checks that outputs meet requirements.

**Language choice:** Go. Schema validation and pattern matching are well-supported, and using the same language as adjacent components simplifies the deployment footprint.

**Alternatives to consider:** This component is not typically standalone; validation logic often embeds in the API layer or Synthesizer. The language should match wherever it lives. If safety scanning requires ML-based content classification, that specific check may call out to a Python service.

#### Validation Checks

| Check | Description | Action on Failure |
|-------|-------------|-------------------|
| Schema compliance | Output matches expected structure | Reject, retry synthesis |
| Completeness | All required fields present | Reject, identify missing subtask |
| Confidence threshold | Aggregated confidence meets minimum | Flag for review or reject |
| Safety scan | No prohibited content in output | Reject, log for review |

---

### Component 7: Resource Budget Manager (Go)

The Budget Manager tracks resource consumption against limits and enforces constraints.

**Language choice:** Go. Counter aggregation, threshold monitoring, and circuit breaker patterns are well-supported with low overhead. Go's simplicity keeps this component lightweight.

**Alternatives to consider:** This is often implemented as a library rather than a standalone service, embedded in the Coordinator or other components. In that case, use the host component's language. For distributed deployments, consider a sidecar pattern with the language matching your service mesh.

#### What It Tracks

| Resource | Tracking Method | Constraint Action |
|----------|-----------------|-------------------|
| Tokens | Sum across all sub-agent calls | Warn at 80%, hard stop at limit |
| Cost | Pricing × token usage | Warn at 80%, hard stop at limit |
| Latency | Wall-clock time from request start | Trigger early synthesis if approaching SLA |
| API calls | Count per external service | Rate limit, queue, or reject |

#### Circuit Breakers

If a workflow is consuming resources at an unexpected rate (runaway decomposition, retry loops), the Budget Manager can trigger circuit breakers. When cost rate exceeds three times the expected rate, the workflow pauses, the operator is alerted, and the system awaits manual approval or timeout before proceeding.

---

### Component 8: Adaptive Learning System (Python)

The Learning System observes outcomes and improves routing, decomposition, and coordination over time.

**Language choice:** Python. The ML ecosystem (PyTorch, scikit-learn, pandas, statistical libraries) lives in Python. There is no viable alternative for ML workloads at this level of sophistication.

**Alternatives to consider:** Scala/Spark for big data scale if telemetry volume exceeds what single-node Python can handle. Julia for numerical computing in specialized cases. We do not recommend Go or Rust for this component because the ML ecosystem gap is too significant.

#### Why Separate?

The Learning System operates on a **batch cadence**, not inline with requests. It ingests telemetry, analyzes patterns, and periodically updates policies. This separation keeps request latency predictable while enabling sophisticated ML. Go and Rust handle request-path performance; Python handles offline learning.

#### What It Learns

| Learning Target | Input Signals | Output |
|-----------------|---------------|--------|
| Decomposition patterns | Task types, template success rates, failure modes | Updated template rankings, new template suggestions |
| Routing optimization | Sub-agent latency/accuracy/cost per task type | Updated CLASSic weights, sub-agent rankings |
| Failure recovery | Failure types, recovery attempts, outcomes | Updated Coordinator retry/fallback policies |
| Elicitation patterns | User clarification interactions, final task understanding | Improved workflow discovery questions |

#### Cross-Enclave Learning: Patterns Only

In environments with security boundaries (defense, healthcare), learning operates on **patterns only, never content**:

- ✅ "Decomposition template X succeeded 94% of the time for task type Y"
- ✅ "Sub-agent A has 50ms lower latency than B for classification tasks"
- ❌ "User asked about [classified content]"
- ❌ "Sub-agent returned [PHI data]"

This approach is compliant by design. It offers different tradeoffs than federated learning or differential privacy: simpler to implement and audit, but less mathematically rigorous in its privacy guarantees. For environments where the primary concern is preventing content leakage across trust boundaries rather than statistical privacy, patterns-only learning may be sufficient.

---

## 6. Sub-Agent SDK

The Sub-Agent SDK is a core deliverable-the standard interface that enables ecosystem development.

### What It Defines

| Aspect | Specification |
|--------|---------------|
| Interface contract | Input schema, output schema, capability declaration |
| Context access | Read-only access to relevant context slices |
| Telemetry emission | Required metrics, latency reporting, confidence scores |
| Invocation rules | Sub-agents cannot invoke other sub-agents; only Coordinator can |
| MCP integration | Standard patterns for accessing tools and data via MCP |

### Why This Matters

A well-defined SDK enables:
- Domain experts to build sub-agents without understanding MCI internals
- Sub-agent marketplace where components are reusable across MCI deployments
- Clear contracts for testing, validation, and certification
- Ecosystem growth independent of any single vendor

---

## 7. Workflow Discovery: Learning Decomposition Patterns

A critical question: where do decomposition templates come from?

### Position: Hybrid with Active Elicitation

Workflow discovery is human-guided but machine-assisted. Rather than asking users to validate proposed decompositions once, the system engages in **iterative clarification**-asking the same underlying question in multiple formulations to reduce noise and surface true intent.

Example elicitation sequence for a task "analyze this contract":

1. "Does this require extracting specific clauses, or understanding overall risk?"
2. "If I found concerning terms, should I flag them or summarize the whole document?"
3. "Would a clause-by-clause breakdown be more useful than a risk summary?"

Each question probes the same underlying need (granularity of analysis) but from different angles. The pattern of responses reveals true intent more reliably than a single question.

### How This Reduces Fragility

Single-question validation produces brittle patterns:
- User says "yes, decompose into clauses"
- Template hardcodes clause extraction
- Different user with similar request needed risk summary
- Template fails

Multi-formulation elicitation builds durable patterns:
- System learns that "contract analysis" has two common intents
- Template includes conditional logic based on elicited signals
- Both use cases succeed

The Adaptive Learning System observes which elicitation sequences yield the most durable templates and refines the questioning approach itself.

---

## 8. Architectural Precedent: From Microservices to Control Planes

The pattern MCI follows is not new. It mirrors the evolution that transformed enterprise software over the past two decades.

**Phase 1: Decomposition (Microservices)**

In the 2000s, enterprises learned that monolithic applications couldn't scale. The solution was decomposition: break the monolith into specialized services, each doing one thing well, communicating through defined interfaces. This enabled independent scaling, independent deployment, technology diversity, and failure isolation.

**Phase 2: Orchestration (Kubernetes)**

Decomposition created a new problem: managing hundreds of services across thousands of containers. The solution was a control plane, Kubernetes, that handled scheduling, scaling, self-healing, and service discovery. The control plane abstracted infrastructure complexity, letting developers focus on services rather than servers.

**AI is following the same path:**

| Era | Software | AI |
|-----|----------|-----|
| Monolith | Single binary, uniform scaling | Single frontier model for everything |
| Decomposition | Microservices, specialized components | Specialized models, sub-agents |
| Control Plane | Kubernetes orchestrates containers | MCI orchestrates models |

**What MCI learns from Kubernetes:**

- **Declarative intent.** Kubernetes manages desired state, not imperative commands. MCI manages task intent, not explicit model calls.
- **Scheduling intelligence.** Kubernetes places workloads based on resource requirements and constraints. MCI routes tasks based on SCALE and CLASSic assessments.
- **Self-healing.** Kubernetes restarts failed containers and reschedules workloads. MCI retries failed sub-agents and routes to alternatives.
- **Observability.** Kubernetes exposes metrics, logs, and traces. MCI emits telemetry for learning and debugging.
- **Abstraction.** Kubernetes abstracts infrastructure from applications. MCI abstracts sub-agent selection from applications.

Organizations that survived the microservices transition, and then the Kubernetes transition, know how to operate MCI systems. The patterns are familiar: decompose by capability, orchestrate through a control plane, observe everything, handle failures gracefully, optimize continuously.

---

## 9. Domain Vignettes: MCI Across Industries

The patterns MCI describes apply wherever organizations face constraints on cost, compliance, latency, or data boundaries. The following illustrative scenarios show how the same architectural decisions manifest in different contexts. These are not case studies-they are thought experiments demonstrating MCI's applicability across domains.

### Vignette 1: Commercial SaaS (Multi-Tenant Cost Optimization)

A B2B SaaS platform provides AI-powered document analysis to thousands of customers. Their challenge: AI costs are eating margin, but customers expect instant results.

**The Problem.** Every document, from a two-page invoice to a 200-page contract, routes through the same frontier model. Simple classification tasks cost the same as complex legal analysis. Customers on the $50/month tier consume the same inference as enterprise customers paying $5,000/month.

**MCI Applied.**

The Workflow Orchestrator applies SCALE to incoming documents. Structure assessment reveals that most documents decompose naturally: extract metadata, classify type, route specialized analysis. Experience data shows that 73% of documents are routine types (invoices, receipts, simple agreements) where specialized sub-agents match frontier accuracy.

The Intelligent Router's compliance gate is straightforward here-all sub-agents meet SOC2, all data stays in the platform's cloud. CLASSic optimization focuses on Cost and Latency, with Accuracy thresholds per document type.

The Adaptive Learning System discovers that certain customer segments (legal, healthcare) have higher accuracy requirements. It learns to weight Accuracy higher for those tenant profiles without explicit configuration.

**Illustrative Impact.** In this scenario, inference costs could drop significantly (potentially 50-70%) while latency improves substantially. The margin unlocked funds further model specialization, creating a flywheel effect.

**Key MCI Contribution.** SCALE's Experience dimension prevented premature optimization-novel document types still route to frontier until pattern confidence builds.

---

### Vignette 2: Regulated Commercial (Healthcare Claims Processing)

A healthcare payer processes millions of claims monthly. AI could accelerate adjudication, but HIPAA compliance and accuracy requirements create hard constraints.

**The Problem.** Claims contain PHI that cannot leave approved environments. Different claim types require different expertise (pharmacy, surgical, behavioral health). Incorrect adjudication creates regulatory exposure and patient harm. The payer operates in multiple states with varying regulations.

**MCI Applied.**

The compliance gate dominates routing decisions. Before any performance optimization, the Router filters sub-agents by: HIPAA certification status, state-specific regulatory compliance, PHI handling authorization, and data residency requirements. A claim from a California Medicaid patient routes only to sub-agents certified for that specific regulatory intersection.

SCALE's Consequence dimension governs decomposition. Pharmacy claims (high volume, well-understood) decompose freely. Complex surgical claims with potential fraud indicators route atomically to specialized sub-agents-the risk of decomposition error exceeds the efficiency gain.

The Context Manager maintains claim history across the member's lifetime, enabling pattern detection (potential fraud, care gaps) while enforcing strict access boundaries. The Adaptive Learning System shares patterns across the enterprise ("claims with characteristic X have 3x denial rate") without sharing any PHI.

**Illustrative Impact.** Adjudication time could drop from days to hours for routine claims. Compliance audit preparation simplifies dramatically because every routing decision is logged with compliance justification. Accuracy on complex claims improves through specialized sub-agents tuned for specific claim types.

**Key MCI Contribution.** Compliance-gate-then-optimize architecture means the system cannot accidentally route PHI to non-compliant sub-agents, regardless of performance pressure.

---

### Vignette 3: Government (Multi-Classification Intelligence Analysis)

An intelligence organization processes information across classification levels. Analysts need AI assistance, but data cannot cross security boundaries.

**The Problem.** Information exists at Unclassified, Secret, and TS/SCI levels. Analysts working at higher levels need to incorporate lower-level information. AI models at each level have different capabilities based on available training data. Connectivity between enclaves is restricted or nonexistent. Edge deployments (field offices, mobile platforms) have intermittent connectivity and constrained compute.

**MCI Applied.**

MCI deploys hierarchically: local coordinators at edge locations, enclave coordinators at each classification level, and no coordinator that spans levels. Each enclave operates its own complete MCI stack with sub-agents certified for that classification.

The compliance gate is absolute. A task tagged TS/SCI routes only to sub-agents in the TS/SCI enclave. There is no "almost compliant." The Router at each level has visibility only into sub-agents at that level-cross-enclave routing doesn't exist.

Edge deployments run with micro models locally. When connectivity exists, the Coordinator syncs patterns (not content) with the regional coordinator. When connectivity drops, local MCI continues operating on cached patterns and local sub-agents.

The Adaptive Learning System demonstrates its patterns-only constraint most clearly here. The TS/SCI enclave learns that certain analytical patterns have high success rates. That pattern knowledge ("decomposition template X works well for task type Y") propagates to lower enclaves. The underlying content never moves.

**Illustrative Impact.** Analysts at each level get AI assistance appropriate to their environment. Edge analysts maintain capability during disconnected operations. The organization's analytical tradecraft improves globally through pattern sharing without any data spillage.

**Key MCI Contribution.** Cross-enclave learning with patterns-only constraints enables organizational learning that would otherwise require impossible data sharing.

---

### The Common Thread

These vignettes differ in specifics but share architectural needs:

| Need | SaaS | Healthcare | Government |
|------|------|------------|------------|
| Compliance gate | SOC2 | HIPAA + state regs | Classification |
| Consequence sensitivity | Low (cost focus) | High (patient impact) | Extreme (national security) |
| Learning constraints | Tenant isolation | PHI boundaries | Classification boundaries |
| Edge requirements | Minimal | Regional data centers | Disconnected operations |

MCI's value is providing a consistent framework for reasoning about these decisions. The compliance gate is always first. SCALE always governs decomposition. CLASSic always optimizes within constraints. The patterns-only learning constraint always enables cross-boundary improvement.

The specific thresholds, sub-agents, and policies differ. The architecture remains constant.

---

## 10. Architectural Positions Summary

This paper takes specific positions on open questions. These represent our considered judgment, but we invite challenge and refinement.

| Question | Position | Rationale |
|----------|----------|-----------|
| Sub-agent/model relationship | Decoupled; same model can power multiple sub-agents with different deployment properties | Compliance is a deployment property, not a model property |
| Routing targets | Route to sub-agents, not models; CLASSic measures observed sub-agent performance | Sub-agents combine model capability with deployment context |
| Routing architecture | Compliance gate, then CLASSic optimization | Compliance is pass/fail; performance is a gradient |
| Cold-start routing | Start conservative, optimize as data accumulates | Effective first, then efficient |
| Workflow discovery | Hybrid with multi-formulation elicitation | Reduces fragility, captures nuance |
| Cross-enclave learning | Patterns only, never content | Compliant by design, auditable |
| Context persistence | Default indefinite; MCI owns memory | Relearning is wasteful |
| Context representation | Declarative structure, learned content | Debuggable yet flexible |

---

## 11. Related Work and Positioning

MCI exists in a crowded field. This section clarifies what MCI contributes relative to existing work and how it complements rather than competes with available tools.

### The Distinction: SDKs vs. Decision Frameworks

Most prior work in this space provides **SDKs and toolkits**, libraries that give developers primitives for building agent systems. MCI provides an **architectural pattern** with opinionated guidance on how to use those primitives in production environments with real constraints.

Use LangChain, Microsoft Agent Framework, or CrewAI to build your system. Use MCI's rubrics to make operational decisions within that system.

### Agent Frameworks and SDKs

**Microsoft Agent Framework** (October 2025) unifies Semantic Kernel and AutoGen into a single SDK for building, deploying, and managing multi-agent systems. It provides the AIAgent abstraction, orchestration patterns (sequential, concurrent, group chat, handoff), MCP support, and enterprise features like observability and durable execution. MCI is designed to work with, not replace, Microsoft Agent Framework. Where MAF provides the "how to build," MCI contributes the "how to decide": which rubrics for decomposition (SCALE), which evaluation framework for routing (CLASSic), how to structure compliance gates, and how to enable cross-boundary learning.

**LangChain** (2022+) established foundational patterns for chaining LLM calls with tools, introducing abstractions for prompts, memory, and agents. Its influence on the field is substantial. MCI adopts similar composability principles but focuses on architectural patterns and decision rubrics rather than implementation primitives.

**AutoGen** (Microsoft Research, 2023+) pioneered multi-agent conversation patterns with support for human-in-the-loop and learned behaviors. Now unified into Microsoft Agent Framework, its research contributions inform MCI's thinking on multi-agent coordination.

**CrewAI** (2023+) developed role-based agent coordination with task decomposition patterns. MCI's Workflow Orchestrator serves a similar function but applies the SCALE rubric for systematic decomposition decisions rather than role-based heuristics.

**LangGraph** extends LangChain with graph-based workflow definitions. MCI's workflow patterns could be expressed in LangGraph; our contribution is the specific rubrics and policies rather than the graph abstraction itself.

### Model Routing and Serving

**vLLM Semantic Router** (evolved to Signal-Decision, November 2025) demonstrated learned routing between models based on query characteristics. MCI adopts similar routing concepts but adds the compliance gate as a prerequisite phase and structures performance optimization around CLASSic dimensions.

**Ray Serve** provides distributed model serving with autoscaling. MCI operates at a higher abstraction layer-Ray Serve could be an implementation substrate for MCI sub-agents.

**Semantic Kernel** (Microsoft, 2023+) offered plugin-based orchestration with planner capabilities before its unification into Microsoft Agent Framework.

### Evaluation Frameworks

**CLASSic** (Aisera, ICLR 2025) provides the Cost, Latency, Accuracy, Stability, Security rubric that MCI adopts for sub-agent evaluation. We apply CLASSic with specific scoring methods and workflow weight profiles but did not create the framework. Our contribution is positioning CLASSic as a runtime optimization layer that operates after compliance gating-not as a complete routing solution.

**Model benchmarks** (MMLU, HumanEval, MATH, GPQA, SWE-Bench) inform sub-agent design decisions. MCI distinguishes design-time model selection (informed by benchmarks) from runtime sub-agent routing (informed by CLASSic).

### Research on Model Routing

**Leeroo Orchestration of Experts** demonstrated training an orchestrator on benchmark performance for intelligent model routing. This represents static learned routing-pre-computing which model handles which query type based on benchmark runs. MCI's adaptive learning operates at runtime with production feedback, learning across multiple dimensions (not just accuracy) and adapting to observed performance.

### What MCI Contributes

Given this context, MCI's contribution is synthesis and opinion for production environments:

**Original contributions:**
- **SCALE rubric** for decomposition decisions (Structure, Consequence, Accuracy, Latency, Experience) with Consequence as governing override
- **Two-phase routing architecture:** compliance gate (pass/fail) before CLASSic optimization (gradient)
- **Cross-enclave learning constraint:** patterns propagate, content never crosses boundaries
- **Sub-agent/model decoupling rationale:** compliance is a deployment property, enabling the same model in multiple sub-agents with different compliance postures

**Architectural positions:**
- Specific language recommendations per component (Go for orchestration, Rust for routing, Python for learning)
- Integration model with MCP as complementary layer
- Context persistence defaults (indefinite, owned by MCI)
- Cold-start strategy (conservative first, optimize as data accumulates)

**What we deliberately don't provide:**
- An SDK (use Microsoft Agent Framework, LangChain, or others)
- A runtime (deploy on Kubernetes, Ray Serve, or cloud platforms)
- Model recommendations (benchmarks evolve too quickly)

MCI is opinionated where existing frameworks are flexible. For environments where "it depends" isn't acceptable-where auditability, compliance, and mission-criticality constrain choices-MCI provides concrete positions to adopt, adapt, or argue against.

---

## 12. Call for Collaboration

This paper proposes an architecture with specific positions. It is not a complete specification; it is an invitation to collaboration.

### What We've Defined

- SCALE rubric for task decomposition with application mechanics
- CLASSic rubric adoption for sub-agent evaluation with application mechanics
- Eight-component architecture with implementation language recommendations
- Five-layer stack from MCP foundation to application interface
- Positions on key architectural questions with rationale

### What Requires Community Input

- Sub-Agent SDK specification details
- MCP extensions for orchestration primitives
- Benchmark frameworks for comparing MCI implementations
- Reference implementations demonstrating patterns
- Refinement of SCALE and CLASSic dimensions and weightings
- Additional failure modes and recovery strategies

---

## Conclusion

Multi-model orchestration is not new. The industry is converging on this pattern-Microsoft's unification of Semantic Kernel and AutoGen into the Agent Framework (October 2025), continued evolution of LangChain and CrewAI, and growing research on intelligent routing all point the same direction.

What's maturing is our understanding of how to make operational decisions within these systems: when to decompose, how to route, what to optimize for, and how to learn across boundaries. These are the questions MCI attempts to answer.

MCI is not an SDK; use Microsoft Agent Framework, LangChain, or others for implementation. MCI is not a runtime; deploy on Kubernetes, Ray Serve, or cloud platforms. MCI is an architectural pattern with decision rubrics: SCALE for decomposition, CLASSic for routing optimization (after compliance gating), patterns-only for cross-boundary learning.

The value is opinion, not novelty. In environments where "it depends" is an unsatisfying answer, MCI provides concrete positions that can be adopted, adapted, or argued against.

We offer this as a contribution to the community-complementary to existing tools, not competitive with them. The conversation about how to do multi-model orchestration well is just beginning.

---

## References

1. **Model Context Protocol (MCP)**. Anthropic, 2024. https://modelcontextprotocol.io

2. **CLASSic: Cost, Latency, Accuracy, Stability, Security Framework for Enterprise AI Agents**. Aisera. Presented at ICLR 2025 Workshop on Building Trust in LLMs and LLM Applications. https://aisera.com/ai-agents-evaluation/

3. **Signal-Decision Driven Architecture: Reshaping Semantic Routing at Scale**. vLLM Project, November 2025. https://blog.vllm.ai/2025/11/19/signal-decision.html

4. **Introducing Microsoft Agent Framework**. Microsoft, October 2025. https://devblogs.microsoft.com/foundry/introducing-microsoft-agent-framework-the-open-source-engine-for-agentic-ai-apps/

5. **LangChain Documentation**. LangChain, Inc. https://docs.langchain.com

6. **CrewAI Documentation**. CrewAI. https://docs.crewai.com

7. **AI Index Report 2025**. Stanford University Human-Centered Artificial Intelligence. https://aiindex.stanford.edu/report/

---

## About the Author

Paul Puckett is Chief Technology Officer at Clarity Innovations and founder/CEO of Relentless Pursuits Consulting Group. His work focuses on AI architecture, distributed systems, and cyber operations for defense and regulated industries. He previously held senior leadership roles at the National Geospatial-Intelligence Agency and the Army.

For collaboration inquiries: [linkedin.com/in/pbp3](https://linkedin.com/in/pbp3)
