"""kta.solve (see README.md)."""

from kta.solve.enumerate import EnumStats, domain_size, enumerate_space, space_size
from kta.solve.mip import MipOutcome, NotExportable, solve_mip
from kta.solve.records import SOLVE_SUBJECT, SolveRecords, record_solve
from kta.solve.result import Candidate, SolveResult, assignment_key
from kta.solve.select import EXHAUSTIVE_LIMIT, THREADS_VAR, rerank, shape_buckets, solve

__all__ = [
    "EXHAUSTIVE_LIMIT",
    "SOLVE_SUBJECT",
    "THREADS_VAR",
    "Candidate",
    "EnumStats",
    "MipOutcome",
    "NotExportable",
    "SolveRecords",
    "SolveResult",
    "assignment_key",
    "domain_size",
    "enumerate_space",
    "record_solve",
    "rerank",
    "shape_buckets",
    "solve",
    "solve_mip",
    "space_size",
]
