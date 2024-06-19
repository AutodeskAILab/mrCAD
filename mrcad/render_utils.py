import numpy as np
import math
from dataclasses import dataclass


def transform(x, y, grid_size, image_size):
    # invert y first for drawing only
    y = -y

    # with the border, the grid is of size grid_size + 2
    # the center of the grid is at (grid_size + 2) // 2
    # add an offset of 1 to account for the border
    x = x + (grid_size + 2) // 2 + 1
    y = y + (grid_size + 2) // 2 + 1
    x = int(x * image_size / (grid_size + 2))
    y = int(y * image_size / (grid_size + 2))
    return x, y


def get_color():
    color = np.random.rand(3)
    # check that it is above 0.5 for at least 1 channel
    above_half = np.any(color > 0.5)
    # check that the 3 channels are not close to each other (avoid gray)
    diff = np.max(color) - np.min(color)
    sufficient_diff = diff > 0.3
    if above_half and sufficient_diff:
        return color
    else:
        return get_color()


def find_circle(x1, y1, x2, y2, x3, y3):
    # If the first point coincides with the third point, it is a full circle
    if x1 == x3 and y1 == y3:
        # Return the midpoint of the first and second point
        ret = [(x1 + x2) / 2, (y1 + y2) / 2]
        print(ret)
        return ret

    x12 = x1 - x2
    x13 = x1 - x3

    y12 = y1 - y2
    y13 = y1 - y3

    y31 = y3 - y1
    y21 = y2 - y1

    x31 = x3 - x1
    x21 = x2 - x1

    # x1^2 - x3^2
    sx13 = x1**2 - x3**2

    # y1^2 - y3^2
    sy13 = y1**2 - y3**2

    sx21 = x2**2 - x1**2
    sy21 = y2**2 - y1**2

    f = (sx13 * x12 + sy13 * x12 + sx21 * x13 + sy21 * x13) / (
        2 * (y31 * x12 - y21 * x13)
    )
    g = (sx13 * y12 + sy13 * y12 + sx21 * y13 + sy21 * y13) / (
        2 * (x31 * y12 - x21 * y13)
    )

    c = -(x1**2) - y1**2 - 2 * g * x1 - 2 * f * y1

    # Equation of circle: x^2 + y^2 + 2*g*x + 2*f*y + c = 0
    # where the center is (h = -g, k = -f) and radius r
    # as r^2 = h^2 + k^2 - c
    h = -g
    k = -f
    sqr_of_r = h**2 + k**2 - c

    # r is the radius
    r = math.sqrt(sqr_of_r)
    return (h, k), r


class Collinear(Exception):
    pass


class OutOfBoundsError(Exception):
    pass


@dataclass
class RenderConfig:
    grid_size: int = 40
    image_size: int = 1280
    line_thickness: int = 4
