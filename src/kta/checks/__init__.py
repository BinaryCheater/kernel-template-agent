"""kta.checks (see README.md): independent checkers writing Check records."""

from kta.checks.anticheat import check_anticheat
from kta.checks.base import Checked, Checker
from kta.checks.full import SpaceItem, SpaceReport, full_check, verify_space
from kta.checks.numeric import check_numeric
from kta.checks.sanitizer import check_sanitizer
from kta.checks.sass import check_sass

__all__ = [
    "Checked",
    "Checker",
    "SpaceItem",
    "SpaceReport",
    "check_anticheat",
    "check_numeric",
    "check_sanitizer",
    "check_sass",
    "full_check",
    "verify_space",
]
