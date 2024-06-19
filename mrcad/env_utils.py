from enum import Enum


class Role(Enum):
    DESIGNER = 1
    MAKER = 2


class OutOfTurnError(Exception):
    pass
