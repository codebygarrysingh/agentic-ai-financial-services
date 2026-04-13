# Agentic AI for Financial Services

> Multi-agent AI system patterns for financial services: regulatory compliance intelligence, real-time risk monitoring, and autonomous regulatory reporting. Production-hardened for capital markets and regulatory technology.

[![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=flat&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph)
[![LangChain](https://img.shields.io/badge/LangChain_0.3-1C3C3C?style=flat&logo=langchain&logoColor=white)](https://langchain.com)
[![Azure](https://img.shields.io/badge/Azure_AKS-0078D4?style=flat&logo=microsoftazure&logoColor=white)](https://azure.microsoft.com)
[![OpenAI](https://img.shields.io/badge/GPT--4o-412991?style=flat&logo=openai&logoColor=white)](https://openai.com)

---

## Overview

This repository documents production multi-agent AI patterns developed for enterprise financial services — capital markets, regulatory compliance, and government payment systems.

Financial services presents uniquely demanding requirements for agentic AI:
- **Auditability** — every agent decision must be traceable to a source document
- **Determinism** — same input, same output (no hallucination on regulatory facts)
- **Latency** — market data agents must act in sub-second windows
- **Compliance** — outputs must satisfy OSFI, Basel III, and SOC2 constraints

These patterns address all four.

---

## Agent Architectures

### 1. Regulatory Compliance Intelligence Agent

```
                    ┌─────────────────────────────────┐
                    │    COMPLIANCE SUPERVISOR AGENT   │
                    │    (orchestrates sub-agents)     │
                    └──────────────┬──────────────────┘
                                   │
              ┌────────────────────┼───────────────────┐
              ▼                    ▼                   ▼
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │  DOCUMENT       │  │   ANALYSIS      │  │   REPORTING     │
    │  RETRIEVAL      │  │   AGENT         │  │   AGENT         │
    │  AGENT          │  │                 │  │                 │
    │  (RAG + KG)     │  │  (reasoning +   │  │  (structured    │
    │                 │  │   extraction)   │  │   output)       │
    └─────────────────┘  └─────────────────┘  └─────────────────┘
```

**Outcome:** 60% reduction in manual compliance processing at RegCore.AI production deployment.

### 2. Real-Time Risk Monitoring Agent

```
    Market Data Feed (Kafka)
            │
            ▼
    ┌───────────────────┐
    │  EVENT CLASSIFIER │  ← ReAct pattern: observe → reason → act
    │  AGENT            │
    └────────┬──────────┘
             │ triggers
    ┌────────▼──────────┐
    │  RISK CALCULATOR  │
    │  AGENT            │  ← Tool use: VaR, Greeks, scenario analysis
    └────────┬──────────┘
             │
    ┌────────▼──────────┐
    │  ALERT ROUTING    │
    │  AGENT            │  ← Human-in-the-loop for threshold breaches
    └───────────────────┘
```

### 3. Regulatory Reporting Agent (Basel III / OSFI)

```
    Source Systems (GL, Risk, Trading)
            │
            ▼
    ┌───────────────────────────────┐
    │  DATA VALIDATION AGENT        │  ← Great Expectations + custom rules
    └───────────────┬───────────────┘
                    │
    ┌───────────────▼───────────────┐
    │  CALCULATION AGENT            │  ← LCR, NSFR, FR2052A, Basel III
    └───────────────┬───────────────┘
                    │
    ┌───────────────▼───────────────┐
    │  RECONCILIATION AGENT         │  ← Cross-system consistency checks
    └───────────────┬───────────────┘
                    │
    ┌───────────────▼───────────────┐
    │  SUBMISSION AGENT             │  ← Formatted regulatory output + audit trail
    └───────────────────────────────┘
```

**Outcome:** 80% reduction in report processing time at RBC.

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent Framework | LangGraph (state machine), LangChain 0.3 |
| LLMs | GPT-4o (reasoning), Claude 3.5 Sonnet (extraction), Gemini (classification) |
| Knowledge Graph | Neo4j (regulatory entity relationships) |
| Vector Store | Pinecone (regulatory document corpus) |
| Streaming | Apache Kafka (market data events) |
| State Management | Redis (agent state, conversation memory) |
| Deployment | Azure AKS, Docker, Kubernetes |
| Observability | LangSmith, OpenTelemetry, Azure Monitor |
| Human-in-Loop | Custom approval workflow + Slack integration |

---

## Project Structure

```
agentic-ai-financial-services/
├── src/
│   ├── agents/
│   │   ├── compliance/
│   │   │   ├── supervisor_agent.py        # Orchestration with LangGraph
│   │   │   ├── document_retrieval_agent.py # RAG + knowledge graph
│   │   │   └── analysis_agent.py           # Structured extraction
│   │   ├── risk/
│   │   │   ├── event_classifier_agent.py   # ReAct pattern
│   │   │   ├── risk_calculator_agent.py    # Tool-augmented reasoning
│   │   │   └── alert_router_agent.py       # HITL escalation
│   │   └── reporting/
│   │       ├── validation_agent.py         # Data quality gates
│   │       ├── calculation_agent.py        # Regulatory formulas
│   │       └── submission_agent.py         # Formatted output + audit
│   ├── tools/
│   │   ├── regulatory_search.py            # Domain-specific retrieval
│   │   ├── risk_calculators.py             # VaR, Greeks, LCR, NSFR
│   │   └── knowledge_graph.py              # Neo4j entity lookup
│   ├── memory/
│   │   ├── conversation_memory.py          # Redis-backed agent memory
│   │   └── entity_memory.py               # Cross-session entity tracking
│   └── guardrails/
│       ├── financial_constraints.py        # Domain-specific validation
│       └── audit_trail.py                 # Immutable decision log
├── graphs/
│   ├── compliance_graph.py                # LangGraph state machine
│   └── risk_graph.py
└── requirements.txt
```

---

## Core Pattern: LangGraph Supervisor

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Define the multi-agent state machine
workflow = StateGraph(ComplianceState)

workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("retrieval", document_retrieval_agent)
workflow.add_node("analysis", analysis_agent)
workflow.add_node("reporting", reporting_agent)
workflow.add_node("human_review", human_in_the_loop)

# Conditional routing based on confidence scores
workflow.add_conditional_edges(
    "analysis",
    route_by_confidence,
    {
        "high_confidence": "reporting",
        "low_confidence": "human_review",
        "escalate": END,
    }
)

workflow.set_entry_point("supervisor")
app = workflow.compile(checkpointer=redis_checkpointer)
```

---

## Production Learnings

1. **Supervisor agents beat pure ReAct for multi-step financial workflows** — explicit state machines are more debuggable and auditable than freeform agent chains
2. **Tool call retries need domain-specific backoff** — financial APIs have rate limits and SLAs that differ from general-purpose retry strategies
3. **Human-in-the-loop is not optional for regulatory outputs** — build the escalation path before deployment, not after
4. **Knowledge graphs complement vector search** — entities (regulations, instruments, counterparties) have structured relationships that embeddings alone can't capture

---

## Related Work

- [production-rag-pipeline](https://github.com/codebygarrysingh/production-rag-pipeline) — RAG foundation used by document retrieval agents
- [llmops-reference-architecture](https://github.com/codebygarrysingh/llmops-reference-architecture) — Deployment and monitoring framework
- [real-time-data-platform](https://github.com/codebygarrysingh/real-time-data-platform) — Kafka streaming layer for market data agents

---

## Author

**Garry Singh** — Principal AI & Data Engineer · MSc Oxford · 10+ years financial services

[Portfolio](https://garrysingh.dev) · [LinkedIn](https://linkedin.com/in/singhgarry) · [Book a Consultation](https://calendly.com/garry-singh2902)
