# Model Context Intelligence (MCI)

An architectural pattern for intelligent multi-model orchestration.

## The Problem

Frontier model inference depends on concentrated infrastructure: GPUs from one company, fabricated by one foundry, with packaging capacity constrained through 2026. Organizations route every task through these same models, creating correlated failure modes and economic inefficiency.

Specialized models under 3B parameters now match frontier performance on bounded tasks at 10-50x lower cost. The capability exists. What's missing is the intelligence to use it.

Orchestration exists. Load balancers route traffic. API gateways manage requests. But these are mechanical orchestrators: they route based on static rules, not task semantics.

## The Pattern

MCI defines an architectural pattern for intelligent orchestration with two core decision rubrics:

### SCALE (Decomposition)

When should a task be broken into subtasks?

| Dimension | Question |
|-----------|----------|
| **S**tructure | Is this task naturally decomposable? |
| **C**onsequence | What happens if we fail? |
| **A**ccuracy | What error tolerance exists? |
| **L**atency | What are the time constraints? |
| **E**xperience | How proven is this pattern? |

Consequence governs. Life-safety tasks route to the highest-accuracy option regardless of cost.

### Two-Phase Routing

Where should each task go?

1. **Compliance gate** (pass/fail): Filter to sub-agents meeting regulatory and policy requirements
2. **CLASSic optimization** (gradient): Among compliant options, score on Cost, Latency, Accuracy, Stability, Security

Compliance is non-negotiable. Performance is a gradient within compliant options.

### Patterns-Only Learning

How do we improve across security boundaries?

Learning operates on patterns, never content:
- ✅ "Template X succeeded 94% for task type Y"
- ✅ "Sub-agent A has 50ms lower latency than B"
- ❌ Actual task content or outputs

Patterns propagate across enclaves. Content stays put.

## Documentation

📄 **[Whitepaper: Intelligent Orchestration for the Multi-Model Era](docs/MCI_Whitepaper_Puckett.md)** ([PDF](docs/MCI_Whitepaper_Puckett.pdf))

Covers the full architectural pattern: eight components, five layers, implementation guidance, and domain vignettes.

## Implementation

MCI is a pattern, not an SDK. Implement it with:
- [Microsoft Agent Framework](https://github.com/microsoft/agents)
- [LangChain](https://docs.langchain.com)
- [CrewAI](https://docs.crewai.com)

Or build your own. The pattern is what matters.

## Status

Working paper. Positions defined, community input welcome.

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to engage.

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Author

Paul Puckett  
[linkedin.com/in/pbp3](https://linkedin.com/in/pbp3)
