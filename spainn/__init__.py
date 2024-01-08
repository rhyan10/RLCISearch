import logging

logging.addLevelName(
    logging.WARNING, "\033[1;33m%s\033[1;0m" % logging.getLevelName(logging.WARNING)
)
logging.addLevelName(
    logging.CRITICAL, "\033[1;91m%s\033[1;0m" % logging.getLevelName(logging.CRITICAL)
)

from . import asetools, interface
from .calculator import *
from .cli import *
from .loss import *
from .metric import *
from .model import *
from .multidatamodule import *
from .plotting import *
from .spainn import SPAINN
