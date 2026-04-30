"""
Dialectical cross-paper knowledge schema.

After extracting knowledge from individual papers, a dialectical review
compares findings across all papers to:
  - identify consensus (multiple papers agree → high confidence)
  - surface contradictions (papers disagree → flag uncertainty)
  - explain divergence (different conditions, strains, or product types)
  - produce evidence-weighted recommendations
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class EvidenceStrength(str, Enum):
    HIGH = "high"           # 3+ papers agree with consistent values
    MEDIUM = "medium"       # 2 papers agree, or 1 paper with strong data
    LOW = "low"             # Single paper, indirect evidence
    CONFLICTING = "conflicting"   # Papers give contradictory results
    UNCERTAIN = "uncertain"  # Insufficient data to conclude

    def __str__(self) -> str:
        return self.value


class PaperPosition(BaseModel):
    """One paper's stance on a specific topic."""
    paper: str = Field(description="Source paper filename")
    claim: str = Field(description="What this paper says about the topic")
    value: Optional[str] = Field(
        default=None, description="Specific numeric value or range reported"
    )
    experimental_context: Optional[str] = Field(
        default=None, description="Conditions under which this was observed (strain, product, scale)"
    )
    confidence_in_paper: Optional[str] = Field(
        default=None, description="How convincingly this paper supports the claim"
    )


class ConsensusPoint(BaseModel):
    """A finding confirmed across multiple papers."""
    topic: str = Field(description="Short topic label, e.g. 'induction temperature'")
    consensus_claim: str = Field(
        description="The unified statement all supporting papers agree on"
    )
    recommended_value: Optional[str] = Field(
        default=None, description="Best-supported specific value or range"
    )
    supporting_papers: list[PaperPosition] = Field(
        description="How each supporting paper contributes"
    )
    evidence_strength: EvidenceStrength
    applies_to: list[str] = Field(
        default_factory=list,
        description="Products or conditions this consensus applies to"
    )
    practical_implication: str = Field(
        description="What this means concretely for experiment design"
    )


class ConflictPoint(BaseModel):
    """A finding where papers disagree or give different values."""
    topic: str
    positions: list[PaperPosition] = Field(
        description="Each paper's conflicting stance"
    )
    divergence_explanation: str = Field(
        description="Likely reason for the difference (different strains, scales, products, measurement methods)"
    )
    risk_level: str = Field(
        description="high / medium / low — how much does this disagreement matter for experiment design"
    )
    recommended_approach: str = Field(
        description="What to do in light of the conflict; may suggest testing both conditions"
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Specific questions that would resolve this conflict"
    )


class TopicSynthesis(BaseModel):
    """Full dialectical synthesis for one topic area."""
    topic_area: str = Field(
        description="Broad topic, e.g. 'Methanol Induction Control', 'P4H Co-expression', 'Proteolysis Prevention'"
    )
    summary: str = Field(
        description="2-3 sentence overview of what the literature collectively says"
    )
    consensus_points: list[ConsensusPoint] = Field(default_factory=list)
    conflict_points: list[ConflictPoint] = Field(default_factory=list)
    overall_confidence: EvidenceStrength
    key_uncertainties: list[str] = Field(
        default_factory=list,
        description="Remaining unknowns that could affect experimental outcomes"
    )
    actionable_recommendation: str = Field(
        description="Concrete guidance that accounts for both consensus and conflicts"
    )


class DialecticalReview(BaseModel):
    """Complete cross-paper dialectical review of all extracted knowledge."""
    papers_reviewed: list[str]
    review_date: str
    topic_syntheses: list[TopicSynthesis] = Field(default_factory=list)
    overall_summary: str = Field(
        description="High-level picture of the field's state of knowledge"
    )
    highest_confidence_findings: list[str] = Field(
        default_factory=list,
        description="The 5-8 things we know most reliably — safe to base experiment design on"
    )
    most_uncertain_areas: list[str] = Field(
        default_factory=list,
        description="Areas where individual experiments should include controls / parameter sweeps"
    )
