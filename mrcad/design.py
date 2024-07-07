from typing import Union, Tuple
from dataclasses import dataclass
import numpy as np
import mrcad.render_utils as ru
from mrcad.env_utils import ConstraintType
import cv2
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment


@dataclass
class Line:
    control_points: Tuple[Tuple[float, float], Tuple[float, float]]

    def to_json(self):
        return {"type": "line", "control_points": self.control_points}

    def to_image(self, image: np.ndarray, render_config: ru.RenderConfig):
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

        return slope1 == slope2

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
                (x1, y1) == (x3, y3)
                or (x1, y1) == (x4, y4)
                or (x2, y2) == (x3, y3)
                or (x2, y2) == (x4, y4)
            )
        else:
            (x3, y3), (x4, y4), (x5, y5) = other_curve.control_points
            return (
                (x1, y1) == (x3, y3)
                or (x1, y1) == (x5, y5)
                or (x2, y2) == (x3, y3)
                or (x2, y2) == (x5, y5)
            )


@dataclass
class Arc:
    control_points: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]

    def to_json(self):
        return {"type": "arc", "control_points": self.control_points}

    def to_image(self, image: np.ndarray, render_config: ru.RenderConfig):
        # get and transform the 3 points
        pt_start, pt_mid, pt_end = self.control_points
        pt_start = render_config.transform(pt_start[0], pt_start[1])
        pt_mid = render_config.transform(pt_mid[0], pt_mid[1])
        pt_end = render_config.transform(pt_end[0], pt_end[1])
        # check if they are colinear, if they are, raise error
        term1 = (pt_end[1] - pt_start[1]) * (pt_mid[0] - pt_start[0])
        term2 = (pt_mid[1] - pt_start[1]) * (pt_end[0] - pt_start[0])
        if (term1 - term2) ** 2 < 1e-3:
            raise ru.Collinear

        center, _ = ru.find_circle(
            pt_start[0], pt_start[1], pt_mid[0], pt_mid[1], pt_end[0], pt_end[1]
        )

        # radius is distance from center to start point
        radius = np.sqrt(
            (pt_start[0] - center[0]) ** 2 + (pt_start[1] - center[1]) ** 2
        )

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
        center1, _ = ru.find_circle(x1, y1, x2, y2, x3, y3)
        if isinstance(other_curve, Arc):
            (x4, y4), (x5, y5), (x6, y6) = other_curve.control_points
            center2, _ = ru.find_circle(x4, y4, x5, y5, x6, y6)
        else:
            (x4, y4), (x5, y5) = other_curve.control_points
            center2 = ((x4 + x5) / 2, (y4 + y5) / 2)
        return center1 == center2

    def meeting_ends(self, other_curve: Union["Line", "Arc"]):
        (x1, y1), (x2, y2), (x3, y3) = self.control_points
        if isinstance(other_curve, Line):
            (x4, y4), (x5, y5) = other_curve.control_points
            return (
                (x1, y1) == (x4, y4)
                or (x1, y1) == (x5, y5)
                or (x3, y3) == (x4, y4)
                or (x3, y3) == (x5, y5)
            )
        else:
            (x4, y4), (x5, y5), (x6, y6) = other_curve.control_points
            return (
                (x1, y1) == (x4, y4)
                or (x1, y1) == (x6, y6)
                or (x3, y3) == (x4, y4)
                or (x3, y3) == (x6, y6)
            )


@dataclass
class Circle:
    control_points: Tuple[Tuple[float, float], Tuple[float, float]]

    def to_json(self):
        return {"type": "circle", "control_points": self.control_points}

    def to_image(self, image: np.ndarray, render_config: ru.RenderConfig):
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
            center2, _ = ru.find_circle(x3, y3, x4, y4, x5, y5)
        return center1 == center2


@dataclass
class Design:
    curves: tuple[Union[Line, Arc, Circle]]

    def to_json(self):
        return {
            "curves": [curve.to_json() for curve in self.curves],
        }

    @classmethod
    def from_json(cls, json_drawing: dict):
        curves = []
        for json_curve in json_drawing.get("curves", []):
            if json_curve["type"] == "line":
                curves.append(Line(json_curve["control_points"]))
            elif json_curve["type"] == "arc":
                curves.append(Arc(json_curve["control_points"]))
            elif json_curve["type"] == "circle":
                curves.append(Circle(json_curve["control_points"]))
            else:
                raise ValueError(f"Unknown curve type: {json_curve['type']}")
        return cls(curves)

    def to_image(
        self,
        render_config: ru.RenderConfig = None,
        flatten: bool = False,
        ignore_out_of_bounds: bool = False,
    ):
        if render_config is None:
            # Use defaults if not specified
            render_config = ru.RenderConfig()

        border_size = render_config.image_size // (render_config.grid_size + 2)

        # create a blank RBG image
        img = np.ones(
            (
                render_config.image_size,
                render_config.image_size,
                3,
            )
        ) * np.array(render_config.get_background_color())

        for curve in self.curves:
            img = curve.to_image(
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
            flatten=True,
            ignore_out_of_bounds=True,
            render_config=ru.RenderConfig(image_size=320),
        )
        arr2 = other_design.to_image(
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

    def get_curve_similarity_matrix(
        self, other_design: "Design", render_config: ru.RenderConfig = None
    ):
        if render_config is None:
            render_config = ru.RenderConfig()
        similarity_matrix = np.zeros((len(self.curves), len(other_design.curves)))
        for i, curve1 in enumerate(self.curves):
            for j, curve2 in enumerate(other_design.curves):
                similarity_matrix[i, j] = curve1.similarity(curve2, render_config)

        return similarity_matrix

    def similarity(
        self,
        other_design: "Design",
        render_config: ru.RenderConfig = None,
        truncate: bool = False,
    ):
        if render_config is None:
            render_config = ru.RenderConfig()

        similarity_matrix = self.get_curve_similarity_matrix(
            other_design, render_config
        )

        row_ind, col_ind = linear_sum_assignment(similarity_matrix, maximize=True)
        element_distances = similarity_matrix[row_ind, col_ind]

        if truncate:
            if len(element_distances) == 0:
                return 0
            else:
                return element_distances.mean()
        else:
            unmatched_element_distances = np.zeros(
                max(len(self.curves), len(other_design.curves)) - len(element_distances)
            )
            element_distances_padded = np.concatenate(
                (element_distances, unmatched_element_distances)
            )

            return element_distances_padded.mean()

    def get_constraints(self):
        constraints = {
            (i, j): set()
            for i in range(len(self.curves))
            for j in range(len(self.curves))
            if j > i
        }
        for i, curve1 in enumerate(self.curves):
            for j, curve2 in enumerate(self.curves):
                if j <= i:
                    continue

                if isinstance(curve1, Line):
                    if isinstance(curve2, Line):
                        if curve1.parallel(curve2):
                            constraints[(i, j)].add(ConstraintType.PARALLEL)
                        elif curve1.perpendicular(curve2):
                            constraints[(i, j)].add(ConstraintType.PERPENDICULAR)
                        elif curve1.meeting_ends(curve2):
                            constraints[(i, j)].add(ConstraintType.MEETING_ENDS)
                    elif isinstance(curve2, Arc):
                        if curve1.meeting_ends(curve2):
                            constraints[(i, j)].add(ConstraintType.MEETING_ENDS)
                    else:
                        pass
                elif isinstance(curve1, Arc):
                    if isinstance(curve2, Line):
                        if curve1.meeting_ends(curve2):
                            constraints[(i, j)].add(ConstraintType.MEETING_ENDS)
                    elif isinstance(curve2, Arc):
                        if curve1.concentric(curve2):
                            constraints[(i, j)].add(ConstraintType.CONCENTRIC)
                    elif isinstance(curve2, Circle):
                        if curve1.concentric(curve2):
                            constraints[(i, j)].add(ConstraintType.CONCENTRIC)
                else:
                    if isinstance(curve2, Arc):
                        if curve1.concentric(curve2):
                            constraints[(i, j)].add(ConstraintType.CONCENTRIC)
                    elif isinstance(curve2, Circle):
                        if curve1.concentric(curve2):
                            constraints[(i, j)].add(ConstraintType.CONCENTRIC)
        return constraints

    @classmethod
    def _constraint_score(cls, design1: "Design", design2: "Design"):
        assert len(design1.curves) >= len(design2.curves)
        constraints1 = design1.get_constraints()
        constraints2 = design2.get_constraints()

        similarity_matrix = design1.get_curve_similarity_matrix(design2)
        row_ind, col_ind = linear_sum_assignment(similarity_matrix, maximize=True)
        mapping = {i: None for i in range(len(design1.curves))}
        for i, j in zip(row_ind, col_ind):
            mapping[i] = j

        intersection = 0
        union = 0
        for i in range(len(design1.curves)):
            for j in range(len(design1.curves)):
                if j <= i:
                    continue

                if mapping[i] is None or mapping[j] is None:
                    union += len(constraints1[(i, j)])
                else:
                    if mapping[i] > mapping[j]:
                        p, q = mapping[j], mapping[i]
                    else:
                        p, q = mapping[i], mapping[j]
                    union += len(constraints1[(i, j)].union(constraints2[(p, q)]))
                    intersection += len(
                        constraints1[(i, j)].intersection(constraints2[(p, q)])
                    )

        return intersection / union

    def constraint_score(self, other_design: "Design"):
        if len(self.curves) < len(other_design.curves):
            return Design._constraint_score(other_design, self)
        else:
            return Design._constraint_score(self, other_design)
