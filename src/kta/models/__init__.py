"""kta.models (see README.md).

Plain functions and a frozen model class; no interface implementation is registered,
because there is one model family so far (decision D240).
"""

from kta.models.bound import bound, machine, min_work, unit_time_us
from kta.models.calibrate import MODEL_SUBJECT, calibrate, load_model, save_model
from kta.models.l1 import EvalEffects, EvalOverlap, EvalStorage, EvalTx, L1Result, price, sol_us
from kta.models.machine import MachineModel
from kta.models.stored import latest_machine, load_arch, load_machine
from kta.models.unit_model import Interaction, NoResidentCta, UnitModel, occupancy

__all__ = [
    "MODEL_SUBJECT",
    "EvalEffects",
    "EvalOverlap",
    "EvalStorage",
    "EvalTx",
    "Interaction",
    "L1Result",
    "MachineModel",
    "NoResidentCta",
    "UnitModel",
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
    "sol_us",
    "unit_time_us",
]
