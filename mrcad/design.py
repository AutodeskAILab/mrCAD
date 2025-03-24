from typing import Union, Tuple, Any, Literal
from typing_extensions import Annotated
from collections.abc import Iterable
from pydantic import (
    BaseModel,
    Field,
)
import numpy as np
from math import isclose, isnan
import mrcad.render_utils as ru
import cv2
from sklearn.neighbors import NearestNeighbors
from copy import deepcopy
from PIL import Image
import base64
from io import BytesIO
import itertools


class Line(BaseModel):
    type: Literal["line"] = "line"
    control_points: Tuple[Tuple[float, float], Tuple[float, float]]

    def render(self, image: np.ndarray, render_config: ru.RenderConfig):
        (x1, y1), (x2, y2) = self.control_points
        x1, y1 = render_config.transform(x1, y1)
        x2, y2 = render_config.transform(x2, y2)

        color = render_config.get_design_color()

        return cv2.line(image, (x1, y1), (x2, y2), color, render_config.line_thickness)

    def similarity(self, other_curve, render_config: ru.RenderConfig):
        if not isinstance(other_curve, Line):
            return 0
        else:
            (x1, y1), (x2, y2) = self.control_points
            (x3, y3), (x4, y4) = other_curve.control_points
            d1 = np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
            d2 = np.sqrt((x2 - x4) ** 2 + (y2 - y4) ** 2)
            d_avg = (d1 + d2) / 2
            return np.exp(-d_avg / render_config.grid_size)

    def parallel(self, other_line: "Line"):
        (x1, y1), (x2, y2) = self.control_points
        (x3, y3), (x4, y4) = other_line.control_points

        if x2 - x1 == 0 or x4 - x3 == 0:
            return x2 - x1 == x4 - x3

        slope1 = (y2 - y1) / (x2 - x1)
        slope2 = (y4 - y3) / (x4 - x3)

        return isclose(slope1, slope2, rel_tol=1e-3)

    def parallel_distance(self, other_line: "Line"):
        if not isinstance(other_line, Line):
            return None

        if not self.parallel(other_line):
            return None

        (x1, y1), (x2, y2) = self.control_points
        (x3, y3), (x4, y4) = other_line.control_points

        dx = x2 - x1
        dy = y2 - y1

        px = x3 - x1
        py = y3 - y1

        t1 = (dx * px + dy * py) / (dx * dx + dy * dy)

        dx = x2 - x1
        dy = y2 - y1

        px = x4 - x1
        py = y4 - y1

        t2 = (dx * px + dy * py) / (dx * dx + dy * dy)

        if not (0 <= t1 <= 1) and not (0 <= t2 <= 1):
            return None

        return np.abs((y2 - y1) * x3 - (x2 - x1) * y3 + x2 * y1 - y2 * x1) / np.sqrt(
            (y2 - y1) ** 2 + (x2 - x1) ** 2
        )

    def perpendicular(self, other_line: "Line"):
        (x1, y1), (x2, y2) = self.control_points
        (x3, y3), (x4, y4) = other_line.control_points

        if x2 - x1 == 0:
            return y4 - y3 == 0
        if x4 - x3 == 0:
            return y2 - y1 == 0

        slope1 = (y2 - y1) / (x2 - x1)
        slope2 = (y4 - y3) / (x4 - x3)

        return slope1 * slope2 == -1

    def meeting_ends(self, other_curve: Union["Line", "Arc"]):
        (x1, y1), (x2, y2) = self.control_points
        if isinstance(other_curve, Line):
            (x3, y3), (x4, y4) = other_curve.control_points

            return (
                np.allclose((x1, y1), (x3, y3), rtol=1e-3)
                or np.allclose((x1, y1), (x4, y4), rtol=1e-3)
                or np.allclose((x2, y2), (x3, y3), rtol=1e-3)
                or np.allclose((x2, y2), (x4, y4), rtol=1e-3)
            )
        else:
            (x3, y3), (x4, y4), (x5, y5) = other_curve.control_points

            return (
                np.allclose((x1, y1), (x3, y3), rtol=1e-3)
                or np.allclose((x1, y1), (x5, y5), rtol=1e-3)
                or np.allclose((x2, y2), (x3, y3), rtol=1e-3)
                or np.allclose((x2, y2), (x5, y5), rtol=1e-3)
            )

    def sample_points_on_curve(self, n_points: int = 10):
        # use x1,y1 and x2,y2 as the two points on the segment
        (x1, y1), (x2, y2) = self.control_points

        # sample n_points points on the line, including the two end points
        # use raw python, no numpy
        curve_pts = []
        for i in range(n_points):
            x = x1 + (((x2 - x1) * i) / (n_points - 1))
            y = y1 + (((y2 - y1) * i) / (n_points - 1))
            curve_pts.append((x, y))
        # assert both end points are the same as the segment's end points

        return curve_pts

    def distance_to_point(self, point):
        """
        Calculate the distance from a line segment to a target point.
        Handles floating-point precision issues using epsilon.
        """
        TOLERANCE = 1e-10  # Small tolerance for floating-point comparisons

        # Points on the segment
        (x1, y1), (x2, y2) = self.control_points

        # Target point
        x0, y0 = point

        # Line coefficients: ax + by + c = 0
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2

        # Denominator for closest point calculation
        denominator = a**2 + b**2

        # Handle degenerate segment (zero length)
        if denominator < TOLERANCE:
            # If the segment is essentially a point, return distance to that point
            return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5

        # Closest point on the infinite line
        closest_x = (b * (b * x0 - a * y0) - a * c) / denominator
        closest_y = (a * (-b * x0 + a * y0) - b * c) / denominator

        # Check if the closest point lies on the segment, with epsilon tolerance
        if (
            min(x1, x2) - TOLERANCE <= closest_x <= max(x1, x2) + TOLERANCE
            and min(y1, y2) - TOLERANCE <= closest_y <= max(y1, y2) + TOLERANCE
        ):
            # Return the distance from the target point to the closest point on the segment
            return ((closest_x - x0) ** 2 + (closest_y - y0) ** 2) ** 0.5
        else:
            # Return the distance to the closest endpoint of the segment
            dist_to_start = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            dist_to_end = ((x2 - x0) ** 2 + (y2 - y0) ** 2) ** 0.5
            return min(dist_to_start, dist_to_end)


class Arc(BaseModel):
    type: Literal["arc"] = "arc"
    control_points: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]

    def render(self, image: np.ndarray, render_config: ru.RenderConfig):
        # get and transform the 3 points
        pt_start, pt_mid, pt_end = self.control_points
        pt_start = render_config.transform(pt_start[0], pt_start[1])
        pt_mid = render_config.transform(pt_mid[0], pt_mid[1])
        pt_end = render_config.transform(pt_end[0], pt_end[1])
        # check if they are colinear, if they are, raise error
        term1 = (pt_end[1] - pt_start[1]) * (pt_mid[0] - pt_start[0])
        term2 = (pt_mid[1] - pt_start[1]) * (pt_end[0] - pt_start[0])
        if (term1 - term2) ** 2 < 1e-3:
            return Line(
                control_points=(self.control_points[0], self.control_points[-1])
            ).render(image, render_config)

        # need to create a dummy arc with the rescaled points to find the center and radius
        center, radius = Arc(control_points=[pt_start, pt_mid, pt_end]).find_circle()

        # get the start angle
        start_angle = np.arctan2(pt_start[1] - center[1], pt_start[0] - center[0])
        # get the end angle
        end_angle = np.arctan2(pt_end[1] - center[1], pt_end[0] - center[0])
        # get the angle that the circle passes through the mid point
        mid_angle = np.arctan2(pt_mid[1] - center[1], pt_mid[0] - center[0])

        # convert the angles to degrees
        start_angle = np.degrees(start_angle)
        end_angle = np.degrees(end_angle)
        mid_angle = np.degrees(mid_angle)
        # convert all angles to be between 0 and 360
        start_angle = start_angle % 360
        end_angle = end_angle % 360
        mid_angle = mid_angle % 360
        # swap start with end if start is greater than end
        if start_angle > end_angle:
            start_angle, end_angle = end_angle, start_angle

        travel_ang1 = end_angle - start_angle
        travel_ang2 = -(360 - travel_ang1)

        # check if start < mid < end, if so, use travel_ang1
        if start_angle < mid_angle < end_angle:
            travel_angle = travel_ang1
        else:
            travel_angle = travel_ang2

        center = (int(center[0]), int(center[1]))
        radius = int(radius)

        color = render_config.get_design_color()

        image = cv2.ellipse(
            image,
            center,
            (radius, radius),
            0,
            start_angle,
            start_angle + travel_angle,
            color,
            render_config.line_thickness,
        )

        return image

    def similarity(self, other_curve, render_config: ru.RenderConfig):
        if not isinstance(other_curve, Arc):
            return 0
        else:
            (x1, y1), (x2, y2), (x3, y3) = self.control_points
            (x4, y4), (x5, y5), (x6, y6) = other_curve.control_points
            d1 = np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2)
            d2 = np.sqrt((x2 - x5) ** 2 + (y2 - y5) ** 2)
            d3 = np.sqrt((x3 - x6) ** 2 + (y3 - y6) ** 2)
            d_avg = (d1 + d2 + d3) / 3
            return np.exp(-d_avg / render_config.grid_size)

    def concentric(self, other_curve: Union["Arc", "Circle"]):
        (x1, y1), (x2, y2), (x3, y3) = self.control_points
        center1, _ = self.find_circle()
        if isinstance(other_curve, Arc):
            (x4, y4), (x5, y5), (x6, y6) = other_curve.control_points
            center2, _ = self.find_circle()
        else:
            (x4, y4), (x5, y5) = other_curve.control_points
            center2 = ((x4 + x5) / 2, (y4 + y5) / 2)
        return np.allclose(center1, center2, rtol=1e-3)

    def meeting_ends(self, other_curve: Union["Line", "Arc"]):
        (x1, y1), (x2, y2), (x3, y3) = self.control_points
        if isinstance(other_curve, Line):
            (x4, y4), (x5, y5) = other_curve.control_points

            return (
                np.allclose((x1, y1), (x4, y4), rtol=1e-3)
                or np.allclose((x1, y1), (x5, y5), rtol=1e-3)
                or np.allclose((x3, y3), (x4, y4), rtol=1e-3)
                or np.allclose((x3, y3), (x5, y5), rtol=1e-3)
            )
        else:
            (x4, y4), (x5, y5), (x6, y6) = other_curve.control_points

            return (
                np.allclose((x1, y1), (x4, y4), rtol=1e-3)
                or np.allclose((x1, y1), (x6, y6), rtol=1e-3)
                or np.allclose((x3, y3), (x4, y4), rtol=1e-3)
                or np.allclose((x3, y3), (x6, y6), rtol=1e-3)
            )

    def find_circle(self):
        (x1, y1), (x2, y2), (x3, y3) = self.control_points

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
        r = np.sqrt(sqr_of_r)
        return (h, k), r

    def get_center(self):
        return self.find_circle()[0]

    def get_radius(self):
        return self.find_circle()[1]

    def canonicalize_angles(self):
        pt_start, pt_mid, pt_end = self.control_points
        center, radius = self.find_circle()

        # get the start angle
        start_angle = np.arctan2(pt_start[1] - center[1], pt_start[0] - center[0])

        # get the end angle
        end_angle = np.arctan2(pt_end[1] - center[1], pt_end[0] - center[0])

        # get the angle that the circle passes through the mid point
        mid_angle = np.arctan2(pt_mid[1] - center[1], pt_mid[0] - center[0])

        # expand the angles to be on the linear number line, e.g. smesmesmesme
        # doing it this way you will always find them in the right, increasing order, despite the discontinuity at 0/2pi
        expanded_list = [("s", start_angle), ("m", mid_angle), ("e", end_angle)]
        expanded_list.append(("s", start_angle + 2 * np.pi))
        expanded_list.append(("m", mid_angle + 2 * np.pi))
        expanded_list.append(("e", end_angle + 2 * np.pi))
        # sort this list by angles
        expanded_list.sort(key=lambda x: x[1])

        order_str = "".join([x[0] for x in expanded_list])

        assert "sme" in order_str or "ems" in order_str, "something is terribly wrong"

        # note the sequence:
        # 1. getting the angles in increasing order from the list
        # 2. centering the mid angle to be halfway between start and end (taking numerical average)
        # 3. capping the angles to be between 0 and 2pi
        # the steps 2 and 3 CANNOT be swapped, otherwise you lose the CW / CCW information
        # e.g. you'll have two arcs sharing start and end, but
        #      one half-moon to the left, one to the right, and you cannot tell them apart
        if "sme" in order_str:
            # grab the angles of sme from the expanded list
            sme_idx = order_str.index("sme")
            new_start_a, new_mid_a, new_end_a = [
                expanded_list[sme_idx + i][1] for i in range(3)
            ]
            # center the new_mid_a to be halfway between new_start_a and new_end_a
            new_mid_a = (new_start_a + new_end_a) / 2
            # cap the angles to be between 0 and 2pi
            new_start_a = new_start_a % (2 * np.pi)
            new_mid_a = new_mid_a % (2 * np.pi)
            new_end_a = new_end_a % (2 * np.pi)
        if "ems" in order_str:
            # grab the angles of sme from the expanded list
            ems_idx = order_str.index("ems")
            # swap the start with end
            new_start_a, new_mid_a, new_end_a = [
                expanded_list[ems_idx + i][1] for i in range(3)
            ]
            # center the new_mid_a to be halfway between new_start_a and new_end_a
            new_mid_a = (new_start_a + new_end_a) / 2
            # cap the angles to be between 0 and 2pi
            new_start_a = new_start_a % (2 * np.pi)
            new_mid_a = new_mid_a % (2 * np.pi)
            new_end_a = new_end_a % (2 * np.pi)

        return new_start_a, new_mid_a, new_end_a

    def sample_points_on_curve(self, n_points: int = 10):
        try:
            arc_center = self.get_center()
        except ZeroDivisionError:
            return Line(
                control_points=(self.control_points[0], self.control_points[-1])
            ).sample_points_on_curve(n_points)
        arc_start = self.control_points[0]

        start_angle, mid_angle, end_angle = self.canonicalize_angles()

        # check if start > end, if so, add 2pi to end
        if start_angle > end_angle:
            end_angle += 2 * np.pi
        # sample n_curve_pts points on the arc
        # use raw python, no numpy
        curve_pts = []
        for i in range(n_points):
            angle = start_angle + (end_angle - start_angle) * i / (n_points - 1)
            x = arc_center[0] + (
                np.cos(angle)
                * np.sqrt(
                    (arc_start[0] - arc_center[0]) ** 2
                    + (arc_start[1] - arc_center[1]) ** 2
                )
            )
            y = arc_center[1] + (
                np.sin(angle)
                * np.sqrt(
                    (arc_start[0] - arc_center[0]) ** 2
                    + (arc_start[1] - arc_center[1]) ** 2
                )
            )
            curve_pts.append((x, y))
        return curve_pts

    def distance_to_point(self, point):
        arc_start, arc_mid, arc_end = self.control_points

        try:
            arc_center = self.get_center()
        except ZeroDivisionError:
            return Line(control_points=(arc_start, arc_end)).distance_to_point(point)

        # get the vector offset from center to start, center to mid, center to end
        v_c_s = (arc_start[0] - arc_center[0], arc_start[1] - arc_center[1])
        v_c_m = (arc_mid[0] - arc_center[0], arc_mid[1] - arc_center[1])
        v_c_e = (arc_end[0] - arc_center[0], arc_end[1] - arc_center[1])

        # get the vector from center to target
        v_c_t = (point[0] - arc_center[0], point[1] - arc_center[1])

        # get the angle of the vectors
        angle_c_s = np.arctan2(v_c_s[1], v_c_s[0])
        angle_c_m = np.arctan2(v_c_m[1], v_c_m[0])
        angle_c_e = np.arctan2(v_c_e[1], v_c_e[0])
        angle_c_t = np.arctan2(v_c_t[1], v_c_t[0])

        expanded_list = [
            ("s", angle_c_s),
            ("m", angle_c_m),
            ("e", angle_c_e),
            ("t", angle_c_t),
        ]
        # add 2pi to everything in the expanded list
        expanded_list_2pi = [(name, angle + 2 * np.pi) for name, angle in expanded_list]
        together = expanded_list + expanded_list_2pi

        # sort the list by angle
        together.sort(key=lambda x: x[1])
        # convert it into a string of letters
        order_str = "".join([x[0] for x in together])
        # replace 'm' and 't' with 'x'
        order_str = order_str.replace("m", "x").replace("t", "x")

        pt_on_arc = "sxxe" in order_str or "exxs" in order_str

        # if the point is on the arc, return the distance from the point to the circle of the arc
        if pt_on_arc:
            radius = (v_c_s[0] ** 2 + v_c_s[1] ** 2) ** 0.5
            dist_target_center = (v_c_t[0] ** 2 + v_c_t[1] ** 2) ** 0.5
            return abs(dist_target_center - radius)
        else:
            # otherwise return the distance from the point to the closest end-point of the arc
            dist_start = (
                (arc_start[0] - point[0]) ** 2 + (arc_start[1] - point[1]) ** 2
            ) ** 0.5
            dist_end = (
                (arc_end[0] - point[0]) ** 2 + (arc_end[1] - point[1]) ** 2
            ) ** 0.5
            return min(dist_start, dist_end)


class Circle(BaseModel):
    type: Literal["circle"] = "circle"
    control_points: Tuple[Tuple[float, float], Tuple[float, float]]

    def render(self, image: np.ndarray, render_config: ru.RenderConfig):
        pt1, pt2 = self.control_points
        pt1 = render_config.transform(pt1[0], pt1[1])
        pt2 = render_config.transform(pt2[0], pt2[1])
        # the center is between pt1 and pt2
        center = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        # the radius is the distance between pt1 and pt2 divided by 2
        radius = int(np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) / 2)

        color = render_config.get_design_color()

        return cv2.circle(image, center, radius, color, render_config.line_thickness)

    def similarity(self, other_curve, render_config: ru.RenderConfig):
        if not isinstance(other_curve, Circle):
            return 0
        else:
            (x1, y1), (x2, y2) = self.control_points
            (x3, y3), (x4, y4) = other_curve.control_points
            c1_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            c1_diameter = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            c2_center = ((x3 + x4) / 2, (y3 + y4) / 2)
            c2_diameter = np.sqrt((x4 - x3) ** 2 + (y4 - y3) ** 2)

            center_dist = np.sqrt(
                (c1_center[0] - c2_center[0]) ** 2 + (c1_center[1] - c2_center[1]) ** 2
            )

            diameter_diff = np.abs(c1_diameter - c2_diameter)

            circle_distance = (center_dist + diameter_diff) / 2

            return np.exp(-circle_distance / render_config.grid_size)

    def concentric(self, other_curve: Union["Arc", "Circle"]):
        (x1, y1), (x2, y2) = self.control_points
        center1 = ((x1 + x2) / 2, (y1 + y2) / 2)
        if isinstance(other_curve, Circle):
            (x3, y3), (x4, y4) = other_curve.control_points
            center2 = ((x3 + x4) / 2, (y3 + y4) / 2)
        else:
            (x3, y3), (x4, y4), (x5, y5) = other_curve.control_points
            center2, _ = self.find_circle()
        return np.allclose(center1, center2, rtol=1e-3)

    def get_center(self):
        (x1, y1), (x2, y2) = self.control_points
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def get_radius(self):
        (x1, y1), (x2, y2) = self.control_points
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) / 2

    def sample_points_on_curve(self, n_points: int = 10):
        center = self.get_center()
        radius = self.get_radius()
        # sample n_curve_pts points on the circle
        # use raw python, no numpy
        curve_pts = []
        for i in range(n_points):
            angle = (2 * np.pi * i) / n_points
            x = center[0] + (radius * np.cos(angle))
            y = center[1] + (radius * np.sin(angle))
            curve_pts.append((x, y))
        return curve_pts

    def distance_to_point(self, point):
        center = self.get_center()
        radius = self.get_radius()

        return np.abs(
            np.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2) - radius
        )


Curve = Annotated[Union[Line, Arc, Circle], Field(discriminator="type")]


class Design(BaseModel):
    curves: Tuple[Curve, ...]

    def render(
        self,
        image: np.ndarray = None,  # Render onto an existing image if provided else initialize a blank image
        render_config: ru.RenderConfig = None,
        flatten: bool = False,
        ignore_out_of_bounds: bool = False,
    ):
        if render_config is None:
            # Use defaults if not specified
            render_config = ru.RenderConfig()

        border_size = render_config.image_size // (render_config.grid_size + 2)

        if image is None:
            # create a blank RBG image
            img = np.ones(
                (
                    render_config.image_size,
                    render_config.image_size,
                    3,
                )
            ) * np.array(render_config.get_background_color())
        else:
            img = deepcopy(image)

        for curve in self.curves:
            img = curve.render(
                img,
                render_config,
            )

        if not ignore_out_of_bounds:
            if (
                img[: border_size - render_config.line_thickness // 2, :, :] != 1
            ).any():
                raise ru.OutOfBoundsError
            if (
                img[-(border_size - render_config.line_thickness // 2) :, :, :] != 1
            ).any():
                raise ru.OutOfBoundsError
            if (
                img[:, : border_size - render_config.line_thickness // 2, :] != 1
            ).any():
                raise ru.OutOfBoundsError
            if (
                img[:, -(border_size - render_config.line_thickness // 2) :, :] != 1
            ).any():
                raise ru.OutOfBoundsError

        if flatten:
            return np.logical_and.reduce(img, axis=-1)
        else:
            return img

    def to_image(
        self,
        return_image_type: str = "PIL.Image",
        image=None,  # Render onto an existing image if provided else initialize a blank image
        render_config: ru.RenderConfig = None,
        flatten: bool = False,
        ignore_out_of_bounds: bool = False,
    ):
        assert return_image_type in [
            "PIL.Image",
            "numpy.ndarray",
            "base64",
        ], f"Unknown return_image_type: {return_image_type}"

        canvas = None
        if image is not None:
            if isinstance(image, np.ndarray):
                if (
                    image.dtype == np.float64
                    and image.max() <= 1.0
                    and image.min() >= 0.0
                ):
                    canvas = image
                elif (
                    image.dtype == np.uint8 and image.max() <= 255 and image.min() >= 0
                ):
                    canvas = image / 255.0
                else:
                    raise ValueError("Invalid image type")
            elif isinstance(image, bytes):
                canvas = np.array(Image.open(BytesIO(image))) / 255.0

        rendered = self.render(
            image=canvas,
            render_config=render_config,
            flatten=flatten,
            ignore_out_of_bounds=ignore_out_of_bounds,
        )

        if return_image_type == "PIL.Image":
            return Image.fromarray((rendered * 255).astype(np.uint8))
        elif return_image_type == "numpy.ndarray":
            return rendered
        elif return_image_type == "base64":
            img = Image.fromarray((rendered * 255).astype(np.uint8))
            with BytesIO() as buffer:
                img.save(buffer, format="PNG")
                img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            return img_b64

    def save_image(self, filename: str, render_config: ru.RenderConfig = None):
        img = self.to_image(render_config=render_config)
        cv2.imwrite(filename, img)

    def chamfer_distance(self, other_design: "Design"):
        """
        Calculate the Chamfer distance between two designs.

        Args:
        - point_cloud1: A NumPy array of points (shape N x 2).
        - point_cloud2: A NumPy array of points (shape M x 2).

        Returns:
        - float: The Chamfer distance.
        """
        arr1 = self.to_image(
            return_image_type="numpy.ndarray",
            flatten=True,
            ignore_out_of_bounds=True,
            render_config=ru.RenderConfig(image_size=320),
        )
        arr2 = other_design.to_image(
            return_image_type="numpy.ndarray",
            flatten=True,
            ignore_out_of_bounds=True,
            render_config=ru.RenderConfig(image_size=320),
        )

        point_cloud1 = np.argwhere(arr1 == 0)
        point_cloud2 = np.argwhere(arr2 == 0)

        if (len(point_cloud1) == 0) or (len(point_cloud2) == 0):
            return float("inf")

        # Calculate distances from each point in cloud1 to the nearest point in cloud2
        nbrs2 = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(point_cloud2)
        distances1, indices1 = nbrs2.kneighbors(point_cloud1)

        # Calculate distances from each point in cloud2 to the nearest point in cloud1
        nbrs1 = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(point_cloud1)
        distances2, indices2 = nbrs1.kneighbors(point_cloud2)

        # Average the distances
        return (distances1.mean() + distances2.mean()) / 2

    def __design_distance_asymmetric(
        self,
        other_design: "Design",
        max_point_distance: float = 10.0,
        n_points: int = 10,
    ):
        """
        Calculate the design distance between two designs.

        Args:
        - other_design: A Design object.

        Returns:
        - float: The design distance.
        """

        self_points = list(
            itertools.chain.from_iterable(
                [curve.sample_points_on_curve(n_points) for curve in self.curves]
            )
        )

        min_dists = list()
        for point in self_points:
            min_dist = max_point_distance
            for curve in other_design.curves:
                min_dist = min(min_dist, curve.distance_to_point(point))

            min_dists.append(min_dist)

        summed_distances = np.sum(min_dists)

        maximum_summed_distance = max_point_distance * len(self_points)

        if len(self_points) == 0:
            return 1
        else:
            proportional_distance = summed_distances / maximum_summed_distance

            if isnan(proportional_distance):
                return 1
            else:
                return proportional_distance

    def design_distance(
        self,
        other_design: "Design",
        max_distance_proportion: float = 0.25,
        render_config: ru.RenderConfig = None,
    ):
        """
        Calculate the design distance between two designs.

        Args:
        - other_design: A Design object.

        Returns:
        - float: The design distance.
        """

        if render_config is None:
            render_config = ru.RenderConfig()

        max_point_distance = render_config.grid_size * max_distance_proportion

        return (
            self.__design_distance_asymmetric(
                other_design, max_point_distance=max_point_distance
            )
            + other_design.__design_distance_asymmetric(
                self, max_point_distance=max_point_distance
            )
        ) / 2

    def round(self, precision: int):
        rounded_curves = list()
        for curve in self.curves:
            rounded_control_points = tuple(
                (
                    round(x, precision) if precision > 0 else int(round(x, precision)),
                    round(y, precision) if precision > 0 else int(round(y, precision)),
                )
                for x, y in curve.control_points
            )
            if isinstance(curve, Line):
                rounded_curve = Line(control_points=rounded_control_points)
            elif isinstance(curve, Arc):
                rounded_curve = Arc(control_points=rounded_control_points)
            elif isinstance(curve, Circle):
                rounded_curve = Circle(control_points=rounded_control_points)

            rounded_curves.append(rounded_curve)

        return Design(curves=tuple(rounded_curves))
