from dataclasses import dataclass
from mrcad.env_utils import Role
from mrcad.design import Design


@dataclass
class Drawing:
    splines: tuple[tuple[float, float]]


@dataclass
class Action:
    role: Role
    instruction: tuple[str, Drawing]
    design: Design
