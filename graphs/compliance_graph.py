"""
Compliance Intelligence Graph — LangGraph StateGraph implementation.

Orchestrates a multi-agent workflow for regulatory document analysis:
  1. Supervisor routes tasks to specialised sub-agents
  2. Document retrieval agent fetches relevant regulatory context
  3. Analysis agent extracts structured compliance findings
  4. Reporting agent generates audit-ready output
  5. Human-in-the-loop escalation for low-confidence findings

Design principle: explicit state machines over freeform ReAct chains.
Regulatory outputs require full auditability — every state transition is logged.
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from src.agents.compliance.analysis_agent import AnalysisAgent
from src.agents.compliance.document_retrieval_agent import DocumentRetrievalAgent
from src.agents.compliance.supervisor_agent import SupervisorAgent
from src.guardrails.audit_trail import AuditTrail

logger = logging.getLogger(__name__)


class AgentNode(str, Enum):
    SUPERVISOR = "supervisor"
    RETRIEVAL = "document_retrieval"
    ANALYSIS = "analysis"
    REPORTING = "reporting"
    HUMAN_REVIEW = "human_review"


class ComplianceState(TypedDict):
    """
    Shared state across all agents in the compliance workflow.

    LangGraph passes this dict through the graph — each node reads
    what it needs and writes its outputs back to shared state.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    retrieved_documents: list[dict[str, Any]]
    analysis_results: dict[str, Any]
    confidence_score: float
    requires_human_review: bool
    audit_log: list[dict[str, Any]]
    final_report: str | None
    error: str | None


class ComplianceGraph:
    """
    Multi-agent compliance intelligence pipeline as a LangGraph StateGraph.

    Routing logic:
        - High confidence (>0.85) → direct to reporting
        - Medium confidence (0.60–0.85) → additional analysis pass
        - Low confidence (<0.60) → human review escalation
        - Error state → human review with full context

    Usage:
        graph = ComplianceGraph()
        app = graph.compile()
        result = app.invoke({"query": "What are the Basel III Tier 1 capital requirements?"})
    """

    CONFIDENCE_HIGH = 0.85
    CONFIDENCE_LOW = 0.60

    def __init__(self):
        self.supervisor = SupervisorAgent()
        self.retrieval_agent = DocumentRetrievalAgent()
        self.analysis_agent = AnalysisAgent()
        self.audit_trail = AuditTrail()
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(ComplianceState)

        # Register nodes
        graph.add_node(AgentNode.SUPERVISOR, self._supervisor_node)
        graph.add_node(AgentNode.RETRIEVAL, self._retrieval_node)
        graph.add_node(AgentNode.ANALYSIS, self._analysis_node)
        graph.add_node(AgentNode.REPORTING, self._reporting_node)
        graph.add_node(AgentNode.HUMAN_REVIEW, self._human_review_node)

        # Entry point
        graph.set_entry_point(AgentNode.SUPERVISOR)

        # Supervisor decides next step
        graph.add_conditional_edges(
            AgentNode.SUPERVISOR,
            self._route_from_supervisor,
            {
                AgentNode.RETRIEVAL: AgentNode.RETRIEVAL,
                AgentNode.HUMAN_REVIEW: AgentNode.HUMAN_REVIEW,
                END: END,
            },
        )

        # After retrieval → analysis
        graph.add_edge(AgentNode.RETRIEVAL, AgentNode.ANALYSIS)

        # After analysis → route by confidence
        graph.add_conditional_edges(
            AgentNode.ANALYSIS,
            self._route_from_analysis,
            {
                AgentNode.REPORTING: AgentNode.REPORTING,
                AgentNode.HUMAN_REVIEW: AgentNode.HUMAN_REVIEW,
            },
        )

        graph.add_edge(AgentNode.REPORTING, END)
        graph.add_edge(AgentNode.HUMAN_REVIEW, END)

        return graph

    def _route_from_supervisor(self, state: ComplianceState) -> str:
        if state.get("error"):
            return AgentNode.HUMAN_REVIEW
        return AgentNode.RETRIEVAL

    def _route_from_analysis(self, state: ComplianceState) -> str:
        confidence = state.get("confidence_score", 0.0)
        if confidence >= self.CONFIDENCE_HIGH and not state.get("requires_human_review"):
            return AgentNode.REPORTING
        logger.info(
            "Routing to human review — confidence=%.2f, requires_review=%s",
            confidence,
            state.get("requires_human_review"),
        )
        return AgentNode.HUMAN_REVIEW

    def _supervisor_node(self, state: ComplianceState) -> dict[str, Any]:
        self.audit_trail.log_transition(AgentNode.SUPERVISOR, state)
        return self.supervisor.run(state)

    def _retrieval_node(self, state: ComplianceState) -> dict[str, Any]:
        self.audit_trail.log_transition(AgentNode.RETRIEVAL, state)
        return self.retrieval_agent.run(state)

    def _analysis_node(self, state: ComplianceState) -> dict[str, Any]:
        self.audit_trail.log_transition(AgentNode.ANALYSIS, state)
        return self.analysis_agent.run(state)

    def _reporting_node(self, state: ComplianceState) -> dict[str, Any]:
        self.audit_trail.log_transition(AgentNode.REPORTING, state)
        report = self._format_report(state)
        return {"final_report": report}

    def _human_review_node(self, state: ComplianceState) -> dict[str, Any]:
        self.audit_trail.log_transition(AgentNode.HUMAN_REVIEW, state)
        logger.warning("Escalating to human review: %s", state.get("query"))
        return {"requires_human_review": True, "final_report": None}

    def _format_report(self, state: ComplianceState) -> str:
        return (
            f"COMPLIANCE ANALYSIS REPORT\n"
            f"{'=' * 60}\n"
            f"Query: {state['query']}\n"
            f"Confidence: {state.get('confidence_score', 0):.1%}\n"
            f"Documents reviewed: {len(state.get('retrieved_documents', []))}\n"
            f"\nFindings:\n{state.get('analysis_results', {}).get('summary', 'N/A')}\n"
        )

    def compile(self, checkpointer=None):
        """Compile the graph, optionally with a persistence checkpointer."""
        return self._graph.compile(checkpointer=checkpointer)
