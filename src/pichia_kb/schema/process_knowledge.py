"""
Higher-level process knowledge schema.

While entities.py captures raw biological/engineering facts,
this module captures the **control logic** and **design principles**
needed to run a fermentation toward a target product outcome.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ControlPriority(str, Enum):
    CRITICAL = "critical"    # Must follow; deviation causes failure
    IMPORTANT = "important"  # Strongly recommended
    ADVISORY = "advisory"    # Helpful but context-dependent


class ProcessStage(str, Enum):
    SEED_CULTURE = "seed_culture"
    GLYCEROL_BATCH = "glycerol_batch"
    GLYCEROL_FED_BATCH = "glycerol_fed_batch"
    METHANOL_TRANSITION = "methanol_transition"
    METHANOL_INDUCTION = "methanol_induction"
    HARVEST = "harvest"
    DOWNSTREAM = "downstream"


class ControlPrinciple(BaseModel):
    """
    A cause-effect-action rule extracted or synthesized from literature.
    These are the core semantics of fermentation control.
    """
    title: str = Field(description="Short name for this principle")
    stage: Optional[ProcessStage] = Field(
        default=None, description="Which fermentation stage this applies to"
    )
    parameter: str = Field(description="The controlled variable, e.g. temperature, DO, pH, methanol feed")
    observation: str = Field(description="What happens / what we observe that triggers this principle")
    mechanism: str = Field(description="The biological/biochemical reason")
    recommendation: str = Field(description="Concrete actionable guidance")
    target_value: Optional[str] = Field(
        default=None, description="Specific set-point or range, e.g. '22–25°C', '>20% DO'"
    )
    consequence_if_ignored: Optional[str] = Field(
        default=None, description="What goes wrong if this is not followed"
    )
    priority: ControlPriority = Field(default=ControlPriority.IMPORTANT)
    applies_to_product: list[str] = Field(
        default_factory=list,
        description="Specific products or product types this applies to (empty = general)"
    )
    sources: list[str] = Field(default_factory=list)


class ProcessStageSpec(BaseModel):
    """
    Specification for one stage of a Pichia fermentation process.
    Describes what to do, how long, and how to know when to move on.
    """
    stage: ProcessStage
    description: str
    carbon_source: str
    typical_duration: Optional[str] = None
    temperature_celsius: Optional[str] = None
    ph: Optional[str] = None
    dissolved_oxygen_percent: Optional[str] = None
    agitation_rpm: Optional[str] = None
    aeration_vvm: Optional[str] = None
    feed_rate: Optional[str] = None
    transition_trigger: Optional[str] = Field(
        default=None,
        description="Condition to advance to next stage, e.g. 'glycerol exhausted (DO spike)', 'OD600 > 200'"
    )
    key_monitoring: list[str] = Field(
        default_factory=list,
        description="Parameters to watch closely in this stage"
    )
    common_problems: list[str] = Field(default_factory=list)
    notes: str = Field(default="")
    sources: list[str] = Field(default_factory=list)


class FermentationProtocol(BaseModel):
    """
    A complete fermentation protocol for producing a specific product.
    Integrates multiple ProcessStageSpecs into a coherent workflow.
    """
    name: str
    target_product: str
    host_strain: str
    expression_vector: str
    stages: list[ProcessStageSpec] = Field(default_factory=list)
    total_duration_hours: Optional[str] = None
    expected_yield: Optional[str] = None
    critical_success_factors: list[str] = Field(
        default_factory=list,
        description="The 3-5 most important things to get right"
    )
    quality_checkpoints: list[str] = Field(
        default_factory=list,
        description="Analytical checks at key timepoints"
    )
    notes: str = Field(default="")
    sources: list[str] = Field(default_factory=list)


class TroubleshootingEntry(BaseModel):
    """
    Problem → root cause → solution mapping from experimental literature.
    """
    problem: str = Field(description="Observable symptom, e.g. 'low titer', 'protein degradation'")
    stage: Optional[ProcessStage] = None
    root_causes: list[str] = Field(
        default_factory=list,
        description="Likely causes in order of probability"
    )
    diagnostic_steps: list[str] = Field(default_factory=list)
    solutions: list[str] = Field(
        default_factory=list,
        description="Actionable fixes in order of recommendation"
    )
    prevention: Optional[str] = None
    sources: list[str] = Field(default_factory=list)


class ProductQualityFactor(BaseModel):
    """
    Links a process parameter to its effect on product structure/modification/activity.
    Core for controlling product quality.
    """
    factor: str = Field(description="Process variable, e.g. temperature, methanol concentration")
    stage: Optional[ProcessStage] = None
    effect_on_structure: Optional[str] = Field(
        default=None, description="Effect on folding, disulfide bonds, aggregation"
    )
    effect_on_modification: Optional[str] = Field(
        default=None, description="Effect on glycosylation, hydroxylation, signal cleavage"
    )
    effect_on_activity: Optional[str] = Field(
        default=None, description="Effect on biological activity or specific activity"
    )
    optimal_range: Optional[str] = None
    notes: str = Field(default="")
    sources: list[str] = Field(default_factory=list)
