"""kta.models (see README.md).

Plain functions and a frozen model class; no interface implementation is registered,
because there is one model family so far (decision D240).
"""

from kta.models.bound import bound, min_work, unit_time_us
from kta.models.calibrate import MODEL_SUBJECT, calibrate, load_model, save_model
from kta.models.unit_model import Interaction, NoResidentCta, UnitModel, occupancy

__all__ = [
    "MODEL_SUBJECT",
    "Interaction",
    "NoResidentCta",
    "UnitModel",
    "bound",
    "calibrate",
    "load_model",
    "min_work",
    "occupancy",
    "save_model",
    "unit_time_us",
]
