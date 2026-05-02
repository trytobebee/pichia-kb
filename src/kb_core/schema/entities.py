"""
Domain schema for Pichia pastoris knowledge base.

Layer 3 (per-paper extracted facts). Design philosophy:

- **Permissive at extraction time.** Required fields are kept to a minimum
  (typically just an identifying ``name``) so partial mentions in the source
  text aren't dropped. Completeness is the job of a later canonicalization
  step (Layer 3.5), not of pydantic validation here.
- **Enums become Optional[str].** The ``ExpressionType`` / ``IntegrationSite``
  / ``SelectionMarker`` / ``SecretionSignal`` / ``GlycosylationType`` /
  ``FermentationMode`` classes below remain as a *reference vocabulary*
  (documentation + future normalization target) but field types use plain
  strings, so an LLM extracting Chinese-language or vendor-specific values
  doesn't get rejected.
- **Every entity carries provenance.** ``EntityProvenance`` adds
  ``sources`` / ``chunk_ids`` / ``raw_mention`` / ``extraction_confidence``
  so downstream dedup, audit, and RAG citation are all possible.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


# ── Reference vocabularies (kept as Enum for documentation/normalization,
#    but Layer 3 fields take plain ``Optional[str]`` to avoid extraction loss) ─

class ExpressionType(str, Enum):
    CONSTITUTIVE = "constitutive"
    INDUCIBLE = "inducible"
    HYBRID = "hybrid"


class IntegrationSite(str, Enum):
    AOX1 = "AOX1"
    AOX2 = "AOX2"
    HIS4 = "HIS4"
    GAP = "GAP"
    OTHER = "other"


class SelectionMarker(str, Enum):
    HIS4 = "HIS4"
    ZEOCIN = "Zeocin"
    G418 = "G418"
    BLASTICIDIN = "Blasticidin"
    OTHER = "other"


class SecretionSignal(str, Enum):
    ALPHA_MF = "alpha-MF"
    PHO1 = "PHO1"
    OST1 = "OST1"
    NONE = "none"
    OTHER = "other"


class GlycosylationType(str, Enum):
    N_LINKED = "N-linked"
    O_LINKED = "O-linked"
    NONE = "none"


class FermentationMode(str, Enum):
    BATCH = "batch"
    FED_BATCH = "fed-batch"
    CONTINUOUS = "continuous"
    METHANOL_INDUCTION = "methanol_induction"


# ── Provenance base ───────────────────────────────────────────────────────────

class EntityProvenance(BaseModel):
    """Traceability fields carried by every Layer 3 entity.

    These are domain-agnostic and should migrate verbatim to the future
    ``core/`` package when the framework is split from the pichia domain.
    """

    # LLMs frequently emit numeric values for fields modeled as strings
    # (``temperature_celsius: 25`` instead of ``"25"``). Coerce silently
    # rather than reject — losing the fact is worse than losing the type.
    model_config = ConfigDict(coerce_numbers_to_str=True)

    sources: list[str] = Field(
        default_factory=list,
        description="Source paper filename(s) where this entity was found.",
    )
    chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk ID(s) the extractor saw — lets us recover the "
                    "original passage that supports each fact.",
    )
    raw_mention: Optional[str] = Field(
        default=None,
        description="Verbatim phrase from the source text the LLM extracted "
                    "this entity from. Useful for deduplication and audit.",
    )
    extraction_confidence: Optional[str] = Field(
        default=None,
        description="'explicit' (stated outright) | 'inferred' (derived).",
    )


# ── Core Domain Entities ──────────────────────────────────────────────────────

class Strain(EntityProvenance):
    """Pichia pastoris (Komagataella phaffii) host strain."""

    name: str = Field(description="Strain designation, e.g. GS115, X-33, CBS7435")
    aliases: list[str] = Field(default_factory=list)
    genotype: Optional[str] = Field(
        default=None, description="Key genetic markers, e.g. his4, Muts, Mut+"
    )
    methanol_utilization: Optional[str] = Field(
        default=None,
        description="Methanol utilization phenotype: Mut+ / Muts / Mut-",
    )
    protease_deficiency: Optional[str] = Field(default=None)
    parent_strain: Optional[str] = Field(default=None)
    copy_number: Optional[str] = Field(
        default=None,
        description="Reported gene copy number, e.g. '10-copy', '1-3 copies'",
    )
    engineering_modifications: list[str] = Field(
        default_factory=list,
        description="Reported edits beyond the parent, e.g. 'Prm1 overexpression'",
    )
    notes: str = Field(default="")


class Promoter(EntityProvenance):
    """Transcriptional promoter used in Pichia expression."""

    name: str = Field(description="e.g. AOX1, GAP, FLD1, PEX8")
    expression_type: Optional[str] = Field(
        default=None,
        description="constitutive / inducible / hybrid (see ExpressionType)",
    )
    inducer: Optional[str] = Field(default=None)
    strength: Optional[str] = Field(default=None)
    notes: str = Field(default="")


class ExpressionVector(EntityProvenance):
    """Plasmid / expression vector for Pichia."""

    name: str = Field(description="Vector name, e.g. pPICZα, pGAPZ, pPIC9K")
    promoter: Optional[str] = Field(default=None)
    secretion_signal: Optional[str] = Field(
        default=None, description="See SecretionSignal for canonical values"
    )
    selection_marker: Optional[str] = Field(
        default=None, description="See SelectionMarker"
    )
    integration_site: Optional[str] = Field(
        default=None, description="See IntegrationSite"
    )
    copy_number_effect: Optional[str] = Field(default=None)
    his4_complementation: Optional[bool] = Field(default=None)
    tag: Optional[str] = Field(
        default=None, description="Affinity/epitope tag, e.g. His6, FLAG"
    )
    notes: str = Field(default="")


class CultureMedium(EntityProvenance):
    """Defined or complex culture medium formulation."""

    name: str = Field(description="Medium designation, e.g. BMMY, BMGY, FM22, BSM")
    composition: Optional[dict[str, str]] = Field(default=None)
    carbon_source: Optional[str] = Field(default=None)
    nitrogen_source: Optional[str] = Field(default=None)
    trace_elements: Optional[dict[str, str]] = Field(default=None)
    ph_range: Optional[str] = Field(default=None)
    purpose: Optional[str] = Field(
        default=None,
        description="Growth phase / induction phase / seed culture, etc.",
    )
    notes: str = Field(default="")


class FermentationConditionFact(EntityProvenance):
    """Process parameters reported for a specific phase or experiment.

    Layer 3 raw fact (was ``FermentationCondition`` before the rename to
    disambiguate from ``domain_knowledge.FermentationConditionGuideline``).
    """

    phase: Optional[str] = Field(
        default=None,
        description="e.g. glycerol batch, glycerol fed-batch, methanol induction",
    )
    mode: Optional[str] = Field(
        default=None, description="See FermentationMode for canonical values"
    )
    temperature_celsius: Optional[str] = Field(default=None)
    ph: Optional[str] = Field(default=None)
    dissolved_oxygen_percent: Optional[str] = Field(default=None)
    agitation_rpm: Optional[str] = Field(default=None)
    aeration_vvm: Optional[str] = Field(default=None)
    feeding_strategy: Optional[str] = Field(default=None)
    duration_hours: Optional[str] = Field(default=None)
    notes: str = Field(default="")


class GlycosylationPattern(EntityProvenance):
    """N- or O-linked glycosylation characteristics."""

    glycosylation_type: Optional[str] = Field(
        default=None, description="See GlycosylationType for canonical values"
    )
    site_motif: Optional[str] = Field(default=None)
    typical_glycan_structure: Optional[str] = Field(default=None)
    impact_on_activity: Optional[str] = Field(default=None)
    engineering_strategies: list[str] = Field(default_factory=list)
    notes: str = Field(default="")


class TargetProduct(EntityProvenance):
    """Target recombinant protein or metabolite to be produced."""

    name: str
    type: Optional[str] = Field(
        default=None,
        description="e.g. enzyme, antibody fragment, VLP, metabolite, collagen",
    )
    gene_source: Optional[str] = Field(default=None)
    codon_optimized: Optional[bool] = Field(default=None)
    molecular_weight_kda: Optional[float] = Field(default=None)
    expected_yield: Optional[str] = Field(default=None)
    desired_modifications: list[str] = Field(default_factory=list)
    activity_assay: Optional[str] = Field(default=None)
    stability_requirements: Optional[str] = Field(default=None)
    notes: str = Field(default="")


class ProcessParameter(EntityProvenance):
    """Quantitative process analytical parameter with context."""

    parameter_name: str = Field(description="e.g. specific methanol consumption rate")
    symbol: Optional[str] = Field(default=None)
    unit: Optional[str] = Field(default=None)
    typical_range: Optional[str] = Field(default=None)
    optimal_value: Optional[str] = Field(default=None)
    effect_on_expression: Optional[str] = Field(default=None)
    measurement_method: Optional[str] = Field(default=None)
    notes: str = Field(default="")


class AnalyticalMethod(EntityProvenance):
    """Method used to characterize product or process."""

    name: str = Field(description="e.g. SDS-PAGE, Western Blot, HPLC-SEC, LC-MS/MS")
    purpose: Optional[str] = Field(default=None)
    sample_type: Optional[str] = Field(default=None)
    key_conditions: Optional[dict[str, str]] = Field(default=None)
    interpretation_notes: Optional[str] = Field(default=None)
    notes: str = Field(default="")


# ── Knowledge Retrieval Primitives ────────────────────────────────────────────

class KnowledgeChunk(BaseModel):
    """A retrievable piece of knowledge from a paper or structured source."""

    chunk_id: str
    source_file: str = Field(description="PDF filename or source identifier")
    source_ref: Optional[str] = Field(default=None)
    section: Optional[str] = Field(default=None)
    content: str = Field(description="The raw or lightly cleaned text chunk")
    entity_types: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """Structured entities extracted from a single document."""

    source_file: str
    source_ref: Optional[str] = None
    strains: list[Strain] = Field(default_factory=list)
    promoters: list[Promoter] = Field(default_factory=list)
    vectors: list[ExpressionVector] = Field(default_factory=list)
    media: list[CultureMedium] = Field(default_factory=list)
    fermentation_conditions: list[FermentationConditionFact] = Field(default_factory=list)
    glycosylation_patterns: list[GlycosylationPattern] = Field(default_factory=list)
    target_products: list[TargetProduct] = Field(default_factory=list)
    process_parameters: list[ProcessParameter] = Field(default_factory=list)
    analytical_methods: list[AnalyticalMethod] = Field(default_factory=list)
    extraction_notes: str = Field(default="")
