import numpy as np
from mrcad.render_utils import RenderConfig
from mrcad.design import Design, Arc, Circle, Line


def canonicalize_circle(circle: Circle):
    center = (
        (circle.control_points[0][0] + circle.control_points[1][0]) / 2,
        (circle.control_points[0][1] + circle.control_points[1][1]) / 2,
    )
    radius = (
        np.sqrt(
            (circle.control_points[0][0] - circle.control_points[1][0]) ** 2
            + (circle.control_points[0][1] - circle.control_points[1][1]) ** 2
        )
        / 2
    )
    return Circle(
        control_points=(
            (center[0] - radius, center[1]),
            (center[0] + radius, center[1]),
        )
    )


def get_design_from_record(record, render_config: RenderConfig = None):
    if render_config is None:
        render_config = RenderConfig()
    points = {
        k: (
            float(v["i"]) - render_config.grid_size // 2,
            float(v["j"]) - render_config.grid_size // 2,
        )
        for k, v in record["points"].items()
    }
    curves_list = []
    for curve in record["curves"]:
        if curve["name"] == "arc_3pt":
            if curve["start"] == curve["end"]:
                curves_list.append(
                    Circle(
                        control_points=(points[curve["start"]], points[curve["mid"]])
                    )
                )
            else:
                curves_list.append(
                    Arc(
                        control_points=(
                            points[curve["start"]],
                            points[curve["mid"]],
                            points[curve["end"]],
                        )
                    )
                )
        elif curve["name"] == "line":
            curves_list.append(
                Line(control_points=(points[curve["pt1"]], points[curve["pt2"]]))
            )
    return Design(curves=curves_list)


def get_strokes_from_record(record, render_config: RenderConfig = None):
    if render_config is None:
        render_config = RenderConfig()
    return tuple(
        tuple(
            (
                point["x"] * render_config.grid_size / render_config.image_size
                - render_config.grid_size // 2,
                point["y"] * render_config.grid_size / render_config.image_size
                - render_config.grid_size // 2,
            )
            for point in spline
        )
        for spline in record
    )


def normalize_curves(design: Design, render_config: RenderConfig = None):
    if render_config is None:
        render_config = RenderConfig()
    # get all the points
    points_list = list()
    for curve in design.curves:
        if isinstance(curve, Circle):
            points_list.extend(canonicalize_circle(curve).control_points)
        else:
            points_list.extend(curve.control_points)
    points = np.array(points_list)
    # get the x_min, x_max, y_min, y_max
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])

    x_center, y_center = (x_max + x_min) / 2, (y_max + y_min) / 2

    max_length = max(x_max - x_min, y_max - y_min)

    normalized_curves = []
    for curve in design.curves:
        normalized_control_points = []
        for point in curve.control_points:
            x = (point[0] - x_center) / max_length * render_config.grid_size
            y = (point[1] - y_center) / max_length * render_config.grid_size

            normalized_control_points.append((x, y))

        if isinstance(curve, Line):
            normalized_curve = Line(control_points=normalized_control_points)
        elif isinstance(curve, Arc):
            normalized_curve = Arc(control_points=normalized_control_points)
        elif isinstance(curve, Circle):
            normalized_curve = Circle(control_points=normalized_control_points)

        normalized_curves.append(normalized_curve)

    return Design(curves=normalized_curves)
