from typing import Union, Tuple
import numpy as np
from dataclasses import dataclass


class Collinear(Exception):
    pass


class OutOfBoundsError(Exception):
    pass


@dataclass
class RenderConfig:
    grid_size: int = 40
    image_size: int = 400
    line_thickness: int = 4
    drawing_thickness: int = 4
    design_color: Union[Tuple[float, float, float], None] = (0, 0, 0)
    drawing_color: Union[Tuple[float, float, float], None] = (1, 0, 0)
    background_color: Tuple[float, float, float] = (1, 1, 1)
    spline_smoothing: float = 0
    spline_resolution: int = 100

    def get_color(self):
        color = np.random.rand(3)
        # check that it is above 0.5 for at least 1 channel
        above_half = np.any(color > 0.5)
        # check that the 3 channels are not close to each other (avoid gray)
        diff = np.max(color) - np.min(color)
        sufficient_diff = diff > 0.3
        if above_half and sufficient_diff:
            return color
        else:
            return self.get_color()

    def get_design_color(self):
        return self.design_color if self.design_color is not None else self.get_color()

    def get_drawing_color(self):
        return (
            self.drawing_color if self.drawing_color is not None else self.get_color()
        )

    def get_background_color(self):
        return self.background_color

    def transform(self, x, y):
        # with the border, the grid is of size grid_size + 2
        # the center of the grid is at (grid_size + 2) // 2
        # add an offset of 1 to account for the border
        x = x + (self.grid_size + 2) // 2
        y = y + (self.grid_size + 2) // 2
        x = int(x * self.image_size / (self.grid_size + 2))
        y = int(y * self.image_size / (self.grid_size + 2))
        return x, y
