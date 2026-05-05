"""Pydantic models for validating LLM JSON responses.

Used across all agents to catch malformed LLM outputs explicitly
instead of silently failing with missing fields.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class QuestionItem(BaseModel):
    """A single generated question from Agent 2."""
    question: str
    follow_up_q: str = ""
    question_type: str = "factual"
    difficulty: str = "moderate"

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError("Question must be at least 10 characters")
        return v.strip()


class AnswerResponse(BaseModel):
    """LLM answer response from Agent 3."""
    answer: str
    follow_up_a: str = ""

    @field_validator("answer")
    @classmethod
    def answer_not_empty(cls, v):
        if not v or len(v.strip()) < 5:
            raise ValueError("Answer must be at least 5 characters")
        return v.strip()


class ValidationScores(BaseModel):
    """Validation scores from Agent 5."""
    faithfulness: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    tone: float = Field(ge=0.0, le=1.0)
    follow_up_quality: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class MultihopValidationScores(ValidationScores):
    """Validation scores with multi-hop criterion."""
    multi_hop_validity: float = Field(ge=0.0, le=1.0, default=0.0)


class BridgeConceptResponse(BaseModel):
    """Bridge concept extraction response from Agent 2.5."""
    bridge_concepts: list[str] = Field(default_factory=list)
    connection_type: str = "none"
    reasoning: str = ""


class MultihopQuestionItem(BaseModel):
    """A single multi-hop question from Agent 2.5."""
    question: str
    follow_up_q: str = ""
    reasoning_type: str = "synthesis"
    difficulty: str = "complex"
    hop_reasoning: str = ""

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v):
        if not v or len(v.strip()) < 15:
            raise ValueError("Multi-hop question must be at least 15 characters")
        return v.strip()


def parse_llm_response(content: str, model_class: type[BaseModel]) -> BaseModel | None:
    """Parse and validate an LLM JSON response against a Pydantic model.

    Returns None if parsing or validation fails.
    """
    import json

    # Strip markdown code fences
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        data = json.loads(text)
        return model_class.model_validate(data)
    except (json.JSONDecodeError, Exception):
        return None


def parse_llm_response_list(
    content: str, item_class: type[BaseModel],
) -> list[BaseModel]:
    """Parse and validate an LLM JSON array response.

    Returns list of validated items. Invalid items are silently dropped.
    """
    import json

    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        data = json.loads(text)
        if not isinstance(data, list):
            return []
        results = []
        for item in data:
            try:
                results.append(item_class.model_validate(item))
            except Exception:
                continue
        return results
    except json.JSONDecodeError:
        return []
