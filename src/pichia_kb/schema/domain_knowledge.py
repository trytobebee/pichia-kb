"""Schema for domain-level knowledge synthesized across all papers."""

from __future__ import annotations
from pydantic import BaseModel, Field

from .dialectical import EvidenceStrength


class TargetProtein(BaseModel):
    name: str                          # e.g. "III型人源化胶原蛋白"
    aliases: list[str] = Field(default_factory=list)
    protein_type: str                  # "full-length" | "fragment" | "human-like"
    collagen_type: str | None = None   # "I" | "II" | "III"
    sequence_origin: str | None = None # "human gene" | "engineered" | "spliced fragment"
    molecular_weight_kda: str | None = None
    key_features: list[str] = Field(default_factory=list)
    hydroxylation_required: bool | None = None
    papers: list[str] = Field(default_factory=list)


class ProductionSubstrate(BaseModel):
    name: str                          # e.g. "甲醇", "甘油"
    role: str                          # "carbon_source" | "inducer" | "co-feed" | "nitrogen_source"
    phase: str | None = None           # "growth" | "induction" | "both"
    typical_concentration: str | None = None
    notes: str | None = None


class FermentationConditionGuideline(BaseModel):
    """Cross-paper synthesized recommendation for a fermentation parameter.

    Distinct from ``entities.FermentationConditionFact``, which is a per-paper
    raw fact. This is Layer 4 guidance assembled across the corpus.
    """

    parameter: str                     # e.g. "诱导温度"
    typical_range: str                 # e.g. "25-30°C"
    optimal_value: str | None = None   # e.g. "25°C"
    effect: str | None = None          # what happens if out of range
    applies_to_phase: str | None = None
    confidence: EvidenceStrength = EvidenceStrength.MEDIUM


class YieldBenchmark(BaseModel):
    protein: str
    expression_system: str             # e.g. "毕赤酵母摇瓶" | "5L发酵罐高密度"
    yield_value: str                   # e.g. "10.3 g/L"
    key_strategies: list[str] = Field(default_factory=list)
    source_paper: str | None = None


class TechnicalChallenge(BaseModel):
    challenge: str                     # e.g. "蛋白降解 / Kex2蛋白酶切割"
    root_cause: str | None = None
    affected_proteins: list[str] = Field(default_factory=list)
    solutions: list[str] = Field(default_factory=list)
    benefit_if_solved: str | None = None   # quantified if possible
    status: str = "partially_solved"   # "unsolved" | "partially_solved" | "solved"
    source_papers: list[str] = Field(default_factory=list)


class Innovation(BaseModel):
    title: str
    description: str
    innovation_type: str               # "genetic" | "process" | "analytical" | "regulatory"
    result: str | None = None          # e.g. "产量提升33.5%"
    transferability: str | None = None # can this be applied to other proteins/processes?
    source_paper: str | None = None


class DomainKnowledge(BaseModel):
    """Cross-paper domain knowledge for Pichia-based collagen production."""

    synthesis_date: str | None = None
    papers_analyzed: list[str] = Field(default_factory=list)

    target_proteins: list[TargetProtein] = Field(default_factory=list)
    production_substrates: list[ProductionSubstrate] = Field(default_factory=list)
    fermentation_conditions: list[FermentationConditionGuideline] = Field(default_factory=list)
    yield_benchmarks: list[YieldBenchmark] = Field(default_factory=list)
    technical_challenges: list[TechnicalChallenge] = Field(default_factory=list)
    innovations: list[Innovation] = Field(default_factory=list)

    field_maturity: str | None = None      # overall assessment
    industrialization_readiness: str | None = None
    key_open_questions: list[str] = Field(default_factory=list)
