from enum import Enum


class Role(str, Enum):
    DESIGNER = "designer"
    MAKER = "maker"


class OutOfTurnError(Exception):
    pass


class ConstraintType(Enum):
    PARALLEL = 1
    PERPENDICULAR = 2
    MEETING_ENDS = 3
    CONCENTRIC = 4
    TANGENT = 5
