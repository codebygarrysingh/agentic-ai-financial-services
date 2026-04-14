"""
Compliance Supervisor Agent — orchestrates sub-agents in the compliance workflow.

Uses a structured output LLM call to determine routing, rather than freeform
ReAct chains. This is deliberate: financial compliance workflows require
predictable, auditable routing — not emergent agent behaviour.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SupervisorDecision(BaseModel):
    """Structured output schema for supervisor routing decisions."""
    next_step: str = Field(
        description="Next agent to invoke: 'retrieval', 'analysis', 'reporting', or 'human_review'"
    )
    reasoning: str = Field(description="Brief explanation of routing decision")
    requires_escalation: bool = Field(
        default=False,
        description="True if query involves novel regulatory territory requiring human judgement"
    )


SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are the supervisor agent for a regulatory compliance intelligence system.
Your role is to analyse incoming compliance queries and route them to the appropriate specialist agent.

Routing rules:
- ALWAYS start with 'retrieval' to gather regulatory context
- Route to 'human_review' if the query involves novel interpretations, enforcement actions, or cross-jurisdictional ambiguity
- Route to 'human_review' if the query contains client-specific advice requests
- Otherwise route to 'analysis' after retrieval

Respond with a structured routing decision."""
    ),
    ("human", "Query: {query}\n\nCurrent state: {state_summary}")
])


class SupervisorAgent:
    """
    Orchestrates the multi-agent compliance workflow.

    Uses structured LLM output (Pydantic schema) to ensure routing decisions
    are always valid and parseable — never relies on freeform text parsing.
    """

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0):
        # temperature=0.0 for deterministic routing in regulated workflows
        llm = ChatOpenAI(model=model, temperature=temperature)
        self.chain = SUPERVISOR_PROMPT | llm.with_structured_output(SupervisorDecision)

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Analyse the query and return routing decision."""
        state_summary = self._summarise_state(state)

        try:
            decision: SupervisorDecision = self.chain.invoke({
                "query": state["query"],
                "state_summary": state_summary,
            })

            logger.info(
                "Supervisor routing: next=%s, escalate=%s — %s",
                decision.next_step,
                decision.requires_escalation,
                decision.reasoning,
            )

            return {
                "requires_human_review": decision.requires_escalation,
                "audit_log": state.get("audit_log", []) + [{
                    "agent": "supervisor",
                    "decision": decision.model_dump(),
                }],
            }

        except Exception as exc:
            logger.error("Supervisor agent error: %s", exc, exc_info=True)
            return {"error": str(exc), "requires_human_review": True}

    def _summarise_state(self, state: dict[str, Any]) -> str:
        docs_count = len(state.get("retrieved_documents", []))
        has_analysis = bool(state.get("analysis_results"))
        return (
            f"Documents retrieved: {docs_count} | "
            f"Analysis complete: {has_analysis} | "
            f"Prior errors: {state.get('error', 'none')}"
        )
