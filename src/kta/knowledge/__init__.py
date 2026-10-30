"""kta.knowledge (see README.md).

Plain functions over the Store; this package registers no interface implementation,
because nothing here has more than one implementation to choose from (decision D200).
"""

from kta.knowledge.chains import validate_chain
from kta.knowledge.coverage import coverage_report
from kta.knowledge.families import load_family, validate_family
from kta.knowledge.patterns import load_patterns, match_pattern
from kta.knowledge.render import family_overview, render_text_knowledge
from kta.knowledge.retrieve import (
    card,
    diff_templates,
    nearest_chains,
    retrieve_families,
    retrieve_templates,
)
from kta.knowledge.scope import reuse_plan, scope_holds
from kta.knowledge.seed import (
    ImportResult,
    SeedError,
    import_hw_catalog,
    import_references,
    import_seed,
    seed_dir,
)
from kta.knowledge.templates import propose_template, validate_template
from kta.knowledge.types import (
    Card,
    CoverageReport,
    FamilyPackage,
    Issue,
    ReusePlan,
    ReuseStep,
    ScopeResult,
    TxDescription,
    errors_of,
)

__all__ = [
    "Card",
    "CoverageReport",
    "FamilyPackage",
    "ImportResult",
    "Issue",
    "ReusePlan",
    "ReuseStep",
    "ScopeResult",
    "SeedError",
    "TxDescription",
    "card",
    "coverage_report",
    "diff_templates",
    "errors_of",
    "family_overview",
    "import_hw_catalog",
    "import_references",
    "import_seed",
    "load_family",
    "load_patterns",
    "match_pattern",
    "nearest_chains",
    "propose_template",
    "render_text_knowledge",
    "retrieve_families",
    "retrieve_templates",
    "reuse_plan",
    "scope_holds",
    "seed_dir",
    "validate_chain",
    "validate_family",
    "validate_template",
]
