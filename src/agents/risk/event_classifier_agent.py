"""
Risk Event Classifier Agent — ReAct pattern for real-time market event triage.

Processes Kafka market data events and classifies them by risk category,
urgency, and required downstream action. Sub-10ms classification target
for high-frequency trading desk monitoring.

Pattern: ReAct (Reason + Act) — agent reasons about the event, selects
a tool to gather additional context, then acts with a classification decision.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class RiskCategory(str, Enum):
    MARKET_RISK = "market_risk"
    CREDIT_RISK = "credit_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    OPERATIONAL_RISK = "operational_risk"
    COMPLIANCE_RISK = "compliance_risk"
    UNKNOWN = "unknown"


class Urgency(str, Enum):
    CRITICAL = "critical"    # Immediate human escalation required
    HIGH = "high"            # Alert within 60 seconds
    MEDIUM = "medium"        # Batch processing acceptable
    LOW = "low"              # Informational only


@dataclass
class EventClassification:
    event_id: str
    risk_category: RiskCategory
    urgency: Urgency
    confidence: float
    reasoning: str
    recommended_action: str
    latency_ms: float = 0.0
    tools_invoked: list[str] = field(default_factory=list)


REACT_PROMPT = PromptTemplate.from_template("""
You are a risk event classifier for a capital markets trading desk.
Classify incoming market events by risk category and urgency.

You have access to the following tools:
{tools}

Use this format:
Thought: analyse the event to determine risk category
Action: [tool name from {tool_names}]
Action Input: [input to tool]
Observation: [tool output]
... (repeat Thought/Action/Observation as needed)
Thought: I have enough information to classify
Final Answer: JSON with keys: risk_category, urgency, confidence (0-1), recommended_action

Event to classify:
{input}

{agent_scratchpad}
""")


class EventClassifierAgent:
    """
    Classifies real-time market events using ReAct pattern.

    Tools available:
    - position_lookup: Get current position exposure for an instrument
    - volatility_check: Get recent volatility regime for an asset class
    - threshold_check: Compare value against risk limits

    Design note: ReAct is appropriate here (vs supervisor pattern) because
    classification requires flexible tool use based on event type, and
    the decision space is well-bounded (fixed output schema).
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        # gpt-4o-mini for cost/latency efficiency on high-frequency classification
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.tools = self._build_tools()
        agent = create_react_agent(self.llm, self.tools, REACT_PROMPT)
        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,
            max_iterations=3,       # Hard cap — classification must be fast
            handle_parsing_errors=True,
        )

    def classify(self, event: dict[str, Any]) -> EventClassification:
        """Classify a market event. Target: <50ms p95."""
        start = time.perf_counter()

        event_description = self._format_event(event)

        try:
            result = self.executor.invoke({"input": event_description})
            classification = self._parse_result(result["output"], event)
        except Exception as exc:
            logger.error("Classification error for event %s: %s", event.get("id"), exc)
            classification = self._fallback_classification(event)

        classification.latency_ms = (time.perf_counter() - start) * 1000
        logger.debug(
            "Classified event %s → %s (%s) in %.1fms",
            event.get("id"), classification.risk_category,
            classification.urgency, classification.latency_ms
        )
        return classification

    def _format_event(self, event: dict[str, Any]) -> str:
        return (
            f"Event ID: {event.get('id', 'unknown')}\n"
            f"Type: {event.get('type', 'unknown')}\n"
            f"Instrument: {event.get('instrument', 'N/A')}\n"
            f"Value: {event.get('value', 'N/A')}\n"
            f"Timestamp: {event.get('timestamp', 'N/A')}\n"
            f"Metadata: {event.get('metadata', {})}"
        )

    def _parse_result(self, output: str, event: dict[str, Any]) -> EventClassification:
        import json
        import re

        # Extract JSON from agent output
        json_match = re.search(r'\{[^{}]+\}', output, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return EventClassification(
                event_id=event.get("id", "unknown"),
                risk_category=RiskCategory(data.get("risk_category", "unknown")),
                urgency=Urgency(data.get("urgency", "medium")),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
                recommended_action=data.get("recommended_action", "monitor"),
            )
        return self._fallback_classification(event)

    def _fallback_classification(self, event: dict[str, Any]) -> EventClassification:
        """Safe fallback when classification fails — escalate to human."""
        return EventClassification(
            event_id=event.get("id", "unknown"),
            risk_category=RiskCategory.UNKNOWN,
            urgency=Urgency.HIGH,  # Conservative: treat unknown as high urgency
            confidence=0.0,
            reasoning="Classification failed — conservative escalation",
            recommended_action="escalate_to_risk_desk",
        )

    def _build_tools(self) -> list[Tool]:
        return [
            Tool(
                name="position_lookup",
                func=self._position_lookup,
                description="Get current position and exposure for an instrument. Input: instrument symbol.",
            ),
            Tool(
                name="volatility_check",
                func=self._volatility_check,
                description="Get recent volatility regime (low/normal/elevated/extreme) for an asset class.",
            ),
            Tool(
                name="threshold_check",
                func=self._threshold_check,
                description="Check if a value breaches risk limits. Input: 'metric_name:value'.",
            ),
        ]

    def _position_lookup(self, instrument: str) -> str:
        # In production: query Redis position cache or risk system API
        return f"Position for {instrument}: [query risk system]"

    def _volatility_check(self, asset_class: str) -> str:
        # In production: query market data service
        return f"Volatility regime for {asset_class}: [query market data]"

    def _threshold_check(self, metric_value: str) -> str:
        # In production: compare against risk limit database
        return f"Threshold check for {metric_value}: [query limits database]"
