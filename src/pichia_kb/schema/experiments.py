"""
ExperimentRun schema — captures a single fermentation experiment as a complete
工艺单 (parameter snapshot + phase parameters + outcome + linked figure curves).

Each paper typically describes multiple discrete experiments (e.g. shake-flask
screening, 5L bioreactor scale-up, methanol feed optimization). Each is one
ExperimentRun with its own goal (what's fixed / varied / observed), construct
configuration, fermentation setup, list of phase parameter blocks, outcome
metrics, and back-references to figures whose curves quantify it.

Storage: data/structured/<paper>.experiments.json (list[ExperimentRun]).
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field

from .entities import EntityProvenance


class ExperimentGoal(BaseModel):
    """What the experiment studies: what's held fixed, what varies, what's measured."""

    model_config = ConfigDict(coerce_numbers_to_str=True)

    summary: str = Field(description="One-sentence purpose of this experiment.")
    fixed_parameters: list[str] = Field(
        default_factory=list,
        description="Parameters held constant, e.g. ['温度=25°C', '菌株=10-Pp-Pr'].",
    )
    varied_parameters: list[str] = Field(
        default_factory=list,
        description="Independent variable(s) being studied, e.g. ['甲醇浓度: 5/10/20 g/L'].",
    )
    observation_targets: list[str] = Field(
        default_factory=list,
        description="Dependent variables observed, e.g. ['hlCOLIII产量', 'OD600'].",
    )


class StrainConstruct(BaseModel):
    """Snapshot of strain + vector + product configuration used in this experiment."""

    model_config = ConfigDict(coerce_numbers_to_str=True)

    host_strain: Optional[str] = Field(default=None, description="e.g. GS115, X-33")
    parent_strain: Optional[str] = Field(
        default=None, description="If this strain was derived from another"
    )
    expression_vector: Optional[str] = Field(default=None, description="e.g. pPIC9K")
    promoters: list[str] = Field(
        default_factory=list,
        description="One or more promoters, e.g. ['AOX1'] or ['AOX1', 'DAS2']",
    )
    signal_peptide: Optional[str] = Field(default=None, description="e.g. MFα, OST1")
    tag: Optional[str] = Field(default=None, description="e.g. 6×His, FLAG")
    selection: Optional[str] = Field(
        default=None, description="Selection method, e.g. 'G418 高拷贝筛选'"
    )
    target_products: list[str] = Field(
        default_factory=list,
        description="Recombinant product name(s); use multiple entries for co-expression, e.g. ['hlCOLIII'] or ['I型胶原 α1链', 'II型胶原 α1链']",
    )
    product_variants: list[str] = Field(
        default_factory=list,
        description="Detailed product variant descriptions parallel to target_products, e.g. ['III型胶原 α1链片段, Gly-X-Y框架, 部分疏水氨基酸替换']",
    )
    copy_number: Optional[str] = Field(default=None, description="e.g. '10 copies'")


class FermentationSetup(BaseModel):
    """Scale + medium + inoculum configuration."""

    model_config = ConfigDict(coerce_numbers_to_str=True)

    scale: Optional[str] = Field(
        default=None, description="e.g. 5L bioreactor, 250mL shake flask"
    )
    initial_medium: Optional[str] = Field(default=None, description="e.g. BSM, BMGY")
    inoculum_percent: Optional[str] = Field(
        default=None, description="e.g. '4%', '10%'"
    )
    inoculum_od600: Optional[str] = Field(default=None)
    seed_culture_notes: Optional[str] = Field(default=None)


class PhaseParams(BaseModel):
    """Process parameters for one stage of the fermentation."""

    model_config = ConfigDict(coerce_numbers_to_str=True)

    phase_name: str = Field(
        description=(
            "Controlled vocabulary: glycerol_batch | glycerol_fed_batch | "
            "starvation | methanol_induction | shake_flask | seed_culture | "
            "harvest | other. Use the closest match; if 'other', explain in notes."
        ),
    )
    duration_hours: Optional[str] = Field(default=None)
    temperature_celsius: Optional[str] = Field(default=None)
    ph: Optional[str] = Field(default=None)
    agitation_rpm: Optional[str] = Field(default=None)
    aeration_vvm: Optional[str] = Field(default=None)
    dissolved_oxygen_percent: Optional[str] = Field(
        default=None, description="DO target, e.g. '>20%'"
    )
    feeding_strategy: Optional[str] = Field(
        default=None,
        description="Carbon-source feed strategy, e.g. 'DO-stat methanol', '50% glycerol + 1.2% PTM1, 18 mL/h/L'",
    )
    notes: Optional[str] = Field(default=None)


class ExperimentOutcome(BaseModel):
    """Reported quantitative outcomes of the experiment."""

    model_config = ConfigDict(coerce_numbers_to_str=True)

    max_yield: Optional[str] = Field(
        default=None, description="Reported peak product titer, e.g. '10.3 g/L'"
    )
    time_to_max_yield_hours: Optional[str] = Field(
        default=None, description="When peak was observed, e.g. '120h'"
    )
    max_wet_cell_weight: Optional[str] = Field(
        default=None, description="Peak biomass (湿菌重 / WCW), e.g. '270 g/L'"
    )
    max_dry_cell_weight: Optional[str] = Field(default=None)
    max_od600: Optional[str] = Field(default=None)
    productivity: Optional[str] = Field(
        default=None, description="e.g. 'mg/L/h' if reported"
    )
    purity: Optional[str] = Field(default=None)
    activity: Optional[str] = Field(
        default=None, description="Reported biological activity if measured"
    )
    other_results: list[str] = Field(
        default_factory=list,
        description="Any other reported numerical results not captured above.",
    )


class ExperimentRun(EntityProvenance):
    """A single fermentation/expression experiment described in a paper.

    Aggregates strain construct + setup + phase params + outcome + figure
    references into one cohesive 工艺快照 — the granularity at which a
    bioprocess engineer reads and reproduces an experiment.
    """

    experiment_id: str = Field(
        description="Paper-local id, e.g. 'wj-exp-01'. Stable for cross-references."
    )
    title: str = Field(description="Short human-readable label.")
    description: Optional[str] = Field(
        default=None, description="Longer plain-text summary if useful."
    )

    model_config = ConfigDict(coerce_numbers_to_str=True, populate_by_name=True)

    goal: ExperimentGoal
    strain_construct: StrainConstruct = Field(
        default_factory=StrainConstruct, alias="construct"
    )
    setup: FermentationSetup = Field(default_factory=FermentationSetup)
    phases: list[PhaseParams] = Field(default_factory=list)
    outcome: ExperimentOutcome = Field(default_factory=ExperimentOutcome)

    linked_figure_ids: list[str] = Field(
        default_factory=list,
        description="FigureData.figure_id values whose curves quantify this experiment.",
    )

    purification_method: Optional[str] = Field(
        default=None,
        description="Downstream purification, e.g. 'Ni-NTA affinity → SEC → lyophilization'",
    )
    analytical_methods: list[str] = Field(
        default_factory=list,
        description="Methods used to measure the outcome, e.g. ['SDS-PAGE', 'BCA', 'qPCR']",
    )

    paper_section: Optional[str] = Field(
        default=None, description="Paper section/chapter where this experiment is described."
    )


class ExperimentLineageEdge(BaseModel):
    """Directed edge: how one experiment derives from / builds on another within a paper."""

    model_config = ConfigDict(coerce_numbers_to_str=True)

    from_id: str = Field(description="Parent experiment_id (the one being built upon).")
    to_id: str = Field(description="Child experiment_id (the new/derived experiment).")
    relation: str = Field(
        description=(
            "Short controlled-vocabulary label. Common values: "
            "derives_from | scales_up | applies_optimum | varies_parameter | "
            "replaces_component | branches | parallel"
        )
    )
    summary: str = Field(
        description="One-sentence Chinese/English description of how the child differs from the parent.",
    )


class PaperExperiments(BaseModel):
    """Container for all experiments extracted from a single paper."""

    source_file: str
    experiments: list[ExperimentRun] = Field(default_factory=list)
    lineage: list[ExperimentLineageEdge] = Field(
        default_factory=list,
        description="Directed edges between experiments capturing the optimization narrative within the paper.",
    )
    extraction_notes: Optional[str] = Field(
        default=None, description="LLM-emitted notes about ambiguities or gaps."
    )
