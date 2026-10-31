"""kta.models (see README.md).

Plain functions and a frozen model class; no interface implementation is registered,
because there is one model family so far (decision D240).
"""

from kta.models.bound import bound, machine, min_work, unit_time_us
from kta.models.calibrate import MODEL_SUBJECT, calibrate, load_model, save_model
from kta.models.l1 import EvalEffects, EvalOverlap, EvalStorage, EvalTx, L1Result, price, sol_us
from kta.models.l2 import L2Result, simulate
from kta.models.l3 import ExternalSassSimulator
from kta.models.machine import MachineModel
from kta.models.stored import latest_machine, load_arch, load_machine
from kta.models.unit_model import Interaction, NoResidentCta, UnitModel, occupancy
from kta.models.whatif import WhatIf, what_if

__all__ = [
    "MODEL_SUBJECT",
    "EvalEffects",
    "EvalOverlap",
    "EvalStorage",
    "EvalTx",
    "ExternalSassSimulator",
    "Interaction",
    "L1Result",
    "L2Result",
    "MachineModel",
    "NoResidentCta",
    "UnitModel",
    "WhatIf",
    "bound",
    "calibrate",
    "latest_machine",
    "load_arch",
    "load_machine",
    "load_model",
    "machine",
    "min_work",
    "occupancy",
    "price",
    "save_model",
    "simulate",
    "sol_us",
    "unit_time_us",
    "what_if",
]
