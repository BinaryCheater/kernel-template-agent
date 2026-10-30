"""Refinement level numbering and the versioned mapping from older numberings (D441).

The current numbering is the research frontend's 1-7 (docs/design/02 §2, D-T7). Records
written before it used the paper's 0-5. Stored records are never rewritten: their bodies
are upcast on read (``kta.core.schemas.upcast``) through a named, versioned mapping, so a
later renumbering adds a mapping instead of changing this one.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

from kta.core.errors import SchemaError

LEVEL_NAMES: Final = MappingProxyType(
    {
        1: "workload",
        2: "graph",
        3: "operators",
        4: "tensor map",
        5: "instructions",
        6: "architecture",
        7: "microarchitecture",
    }
)

# name -> {old level -> current level}. Never edit a published mapping; add a new name.
LEVEL_MAPPINGS: Final = MappingProxyType(
    {
        # The paper's 0 graph, 1 operator transforms, 2 tensor execution mapping,
        # 3 instruction scheduling, 4 system architecture, 5 microarchitecture.
        # Nothing in it maps to 1 (workload): that level did not exist.
        "paper-0to5@1": MappingProxyType({0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7}),
    }
)
# The mapping used by the schema upcasters for every record written before D441.
LEGACY_MAPPING: Final = "paper-0to5@1"


def map_level(old: int, mapping: str = LEGACY_MAPPING) -> int:
    """The current level of ``old`` under ``mapping``; SchemaError when it has none."""
    table = LEVEL_MAPPINGS.get(mapping)
    if table is None:
        raise SchemaError(f"unknown level mapping {mapping!r}")
    if old not in table:
        raise SchemaError(f"level {old!r} is not in mapping {mapping}")
    return table[old]
