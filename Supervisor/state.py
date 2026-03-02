"""
state.py

Defines the SupervisorState TypedDict and all Pydantic schemas used
across the Supervisor workflow (intent classification, validation, etc.).
"""

from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# SupervisorState -- shared state flowing through the LangGraph workflow
# ---------------------------------------------------------------------------

class SupervisorState(TypedDict):
    """Complete state for a single supervisor turn."""

    # -- Input --
    judge_query: str                          # Current judge question
    case_id: str                              # Active case identifier
    uploaded_files: List[str]                 # File paths if documents uploaded

    # -- Conversation Memory --
    conversation_history: List[dict]          # role/content message pairs
    turn_count: int                           # Current conversation turn

    # -- Intent Classification --
    intent: str                               # ocr / summarize / civil_law_rag / case_doc_rag / reason / multi / off_topic
    target_agents: List[str]                  # Which agents to invoke
    classified_query: str                     # Rewritten/clarified query

    # -- Agent Execution --
    agent_results: Dict[str, Any]             # agent_name -> raw output
    agent_errors: Dict[str, str]              # agent_name -> error message

    # -- Validation --
    validation_status: str                    # pass / fail_hallucination / fail_relevance / fail_completeness
    validation_feedback: str                  # Explanation of what failed
    retry_count: int                          # Current retry attempt
    max_retries: int                          # Default 2

    # -- Document Classification --
    document_classifications: List[Dict[str, Any]]  # Classification results per document

    # -- Output --
    merged_response: str                      # Combined response from all agents
    final_response: str                       # Validated, formatted final answer
    sources: List[str]                        # Citations and references


# ---------------------------------------------------------------------------
# Pydantic schemas for LLM structured output
# ---------------------------------------------------------------------------

class IntentClassification(BaseModel):
    """Structured output from the intent classification LLM call."""

    intent: str = Field(
        description=(
            "One of: ocr, summarize, civil_law_rag, case_doc_rag, "
            "reason, multi, off_topic"
        )
    )
    target_agents: List[str] = Field(
        description="List of agents to invoke (e.g. ['civil_law_rag', 'reason'])"
    )
    rewritten_query: str = Field(
        description="Clarified standalone query using conversation context"
    )
    reasoning: str = Field(
        description="Brief explanation of classification decision"
    )


class ValidationResult(BaseModel):
    """Structured output from the output validation LLM call."""

    hallucination_pass: bool = Field(
        description="True if the response is grounded in source material"
    )
    relevance_pass: bool = Field(
        description="True if the response addresses the judge query"
    )
    completeness_pass: bool = Field(
        description="True if all aspects of the query are covered"
    )
    overall_pass: bool = Field(
        description="True only when all three checks pass"
    )
    feedback: str = Field(
        description="What failed and why, used for retry guidance"
    )
