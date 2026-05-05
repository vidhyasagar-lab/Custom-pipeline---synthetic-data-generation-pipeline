"""Persona and query scenario generation for diverse training data.

Implements:
- P1.4: Persona-driven query generation (beginner, expert, impatient, etc.)
- P1.5: Query style (conversational, web-search, formal) and length control
- P3.10: Query distribution control (target percentages per type)

Inspired by RAGAS scenario-based test generation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════════════
# PERSONAS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Persona:
    """A simulated user persona for query generation."""
    name: str
    description: str
    tone_instruction: str
    example_prefix: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description}


# Built-in personas
PERSONAS = [
    Persona(
        name="beginner",
        description="A new user with no background knowledge, asks basic questions",
        tone_instruction="Use simple, casual language. Ask basic 'what is' and 'how does' questions. "
                         "Show uncertainty: 'I'm not sure I understand...', 'Can you explain simply...'",
        example_prefix="Hey, I'm new to this — ",
    ),
    Persona(
        name="expert",
        description="A domain expert who asks detailed, technical questions",
        tone_instruction="Use precise technical language. Ask about edge cases, implementation details, "
                         "and nuances. Questions should be specific and assume prior knowledge.",
        example_prefix="",
    ),
    Persona(
        name="impatient",
        description="A busy user who wants quick, direct answers",
        tone_instruction="Keep questions very short and direct. No pleasantries. "
                         "Examples: 'What's X?', 'How to do Y?', 'TL;DR on Z?'",
        example_prefix="Quick question: ",
    ),
    Persona(
        name="curious_student",
        description="A student exploring a topic, asks 'why' and 'how' questions",
        tone_instruction="Ask exploratory questions that dig into reasoning and connections. "
                         "Use phrases like 'Why does...', 'How is X related to Y?', 'What happens if...'",
        example_prefix="I'm studying this and wondering — ",
    ),
    Persona(
        name="non_native_speaker",
        description="A user whose first language isn't English, uses simpler phrasing",
        tone_instruction="Use simple sentence structure. May make minor grammar variations. "
                         "Ask clear, straightforward questions. Avoid idioms and complex phrasing.",
        example_prefix="Please help me understand: ",
    ),
]

PERSONA_MAP = {p.name: p for p in PERSONAS}


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY STYLES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QueryStyle:
    """A query formulation style."""
    name: str
    instruction: str
    example: str


QUERY_STYLES = [
    QueryStyle(
        name="conversational",
        instruction="Write the question as a natural chat message, like talking to a friend.",
        example="Hey, can you tell me about X?",
    ),
    QueryStyle(
        name="web_search",
        instruction="Write the question as a short search query — keywords only, no full sentences.",
        example="X benefits vs Y comparison",
    ),
    QueryStyle(
        name="formal",
        instruction="Write the question in a professional, formal tone.",
        example="Could you please explain the differences between X and Y?",
    ),
]

STYLE_MAP = {s.name: s for s in QUERY_STYLES}


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY LENGTHS
# ═══════════════════════════════════════════════════════════════════════════════

QUERY_LENGTHS = {
    "short": "Keep the question under 10 words.",
    "medium": "Write a normal-length question (10-20 words).",
    "long": "Write a detailed question with context (20-40 words). Include background or constraints.",
}


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QueryDistribution:
    """Target distribution for question types."""
    factual: float = 0.25
    explanatory: float = 0.20
    procedural: float = 0.15
    comparative: float = 0.15
    inferential: float = 0.10
    scenario: float = 0.05
    multihop: float = 0.10

    def to_dict(self) -> dict:
        return {
            "factual": self.factual,
            "explanatory": self.explanatory,
            "procedural": self.procedural,
            "comparative": self.comparative,
            "inferential": self.inferential,
            "scenario": self.scenario,
            "multihop": self.multihop,
        }

    def get_instruction(self) -> str:
        """Convert distribution to a prompt instruction."""
        parts = []
        for qtype, pct in self.to_dict().items():
            if pct > 0 and qtype != "multihop":
                parts.append(f"- ~{int(pct*100)}% {qtype}")
        return "Target question type distribution:\n" + "\n".join(parts)


DEFAULT_DISTRIBUTION = QueryDistribution()


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QueryScenario:
    """A complete scenario for generating a question.
    Combines persona + style + length + target type.
    """
    persona: Persona
    style: QueryStyle
    length: str  # "short", "medium", "long"
    target_type: str  # "factual", "explanatory", etc.

    def to_prompt_instruction(self) -> str:
        """Generate a combined instruction for the LLM prompt."""
        lines = [
            f"PERSONA: {self.persona.description}",
            f"PERSONA TONE: {self.persona.tone_instruction}",
            f"QUERY STYLE: {self.style.instruction} (Example: \"{self.style.example}\")",
            f"QUERY LENGTH: {QUERY_LENGTHS.get(self.length, QUERY_LENGTHS['medium'])}",
            f"TARGET QUESTION TYPE: {self.target_type}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "persona": self.persona.name,
            "style": self.style.name,
            "length": self.length,
            "target_type": self.target_type,
        }


def generate_scenarios(
    num_scenarios: int,
    distribution: QueryDistribution | None = None,
    personas: list[str] | None = None,
    styles: list[str] | None = None,
    lengths: list[str] | None = None,
) -> list[QueryScenario]:
    """Generate a diverse set of query scenarios.

    Balances across personas, styles, and lengths while following
    the target type distribution.
    """
    dist = distribution or DEFAULT_DISTRIBUTION
    active_personas = [PERSONA_MAP[p] for p in (personas or list(PERSONA_MAP.keys()))
                       if p in PERSONA_MAP]
    active_styles = [STYLE_MAP[s] for s in (styles or list(STYLE_MAP.keys()))
                     if s in STYLE_MAP]
    active_lengths = [l for l in (lengths or list(QUERY_LENGTHS.keys()))
                      if l in QUERY_LENGTHS]

    if not active_personas:
        active_personas = PERSONAS
    if not active_styles:
        active_styles = QUERY_STYLES
    if not active_lengths:
        active_lengths = list(QUERY_LENGTHS.keys())

    # Build type pool based on distribution
    type_pool = []
    dist_dict = dist.to_dict()
    for qtype, pct in dist_dict.items():
        if qtype == "multihop":
            continue  # Multi-hop is handled separately
        count = max(1, round(num_scenarios * pct))
        type_pool.extend([qtype] * count)
    random.shuffle(type_pool)

    # Pad or trim to exact count
    while len(type_pool) < num_scenarios:
        type_pool.append(random.choice(list(dist_dict.keys())))
    type_pool = type_pool[:num_scenarios]

    scenarios = []
    for i, target_type in enumerate(type_pool):
        scenario = QueryScenario(
            persona=active_personas[i % len(active_personas)],
            style=active_styles[i % len(active_styles)],
            length=active_lengths[i % len(active_lengths)],
            target_type=target_type,
        )
        scenarios.append(scenario)

    return scenarios
