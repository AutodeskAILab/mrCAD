from typing import List
import numpy as np
from mrcad.render_utils import RenderConfig
from mrcad.design import Design, Arc, Circle, Line, Curve
from agents.editing_actions import (
    DeletePoint,
    MovePoint,
    MakeCurve,
    MoveCurve,
    RemoveCurve,
    Edit,
    EditExecution,
)


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
            elif curve["mid"] == curve["start"] or curve["mid"] == curve["end"]:
                continue
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


def deduplicate_curves(curves: List[Curve]):
    """
    Deduplicate curves in a list of curves.

    Args:
        curves (List[Curve]): List of curves

    Returns:
        List[Curve]: List of deduplicated curves
    """
    new_curves = []
    for curve in curves:
        if curve not in new_curves:
            new_curves.append(curve)
    return new_curves


def get_extraneous_curves(original: Design, reconstruction: Design):
    """
    Get extraneous curves in the reconstruction that are not present in the original design.

    Args:
        original (Design): Original design
        reconstruction (Design): Reconstructed design

    Returns:
        List[Curve]: List of extraneous curves
    """
    extraneous_curves = []
    for curve in reconstruction.curves:
        if curve not in original.curves:
            extraneous_curves.append(curve)
    return extraneous_curves


def filter_actions(actions: List[Edit], extraneous_curves: List[Curve]):
    """
    Remove actions that create extraneous curves

    Args:
        actions (List[Edit]): List of actions
        extraneous_curves (List[Curve]): List of extraneous curves

    Returns:
        List[Edit]: Filtered list of actions
    """
    filtered_actions = list()
    for a in actions:
        exclude = False
        if isinstance(a, MakeCurve):
            for e in extraneous_curves:
                if e.type == a.type and e.control_points == a.control_points:
                    exclude = True

        if not exclude:
            filtered_actions.append(a)

    return filtered_actions


def get_edit_actions_from_record(
    record: List, context: Design, execution: Design, render_config: RenderConfig = None
):
    if render_config is None:
        render_config = RenderConfig()
    edit_actions = []

    # convert each of the recorded actions to Edit objects
    for a in record:
        if len(a) == 0:
            continue

        if a[0] == "mk_pt":
            continue
        elif a[0] == "del_pt":
            edit_actions.append(
                DeletePoint(
                    point=(
                        a[1] - render_config.grid_size // 2,
                        a[2] - render_config.grid_size // 2,
                    )
                )
            )
        elif a[0] == "mk_curve":
            curve_type, *control_points = a[1]
            if curve_type == "line":
                edit_actions.append(
                    MakeCurve(
                        type=curve_type,
                        control_points=list(
                            map(
                                lambda x: (
                                    x[0] - render_config.grid_size // 2,
                                    x[1] - render_config.grid_size // 2,
                                ),
                                control_points,
                            )
                        ),
                    )
                )
            else:
                if control_points[0] == control_points[-1]:
                    edit_actions.append(
                        MakeCurve(
                            type="circle",
                            control_points=list(
                                map(
                                    lambda x: (
                                        x[0] - render_config.grid_size // 2,
                                        x[1] - render_config.grid_size // 2,
                                    ),
                                    [control_points[0], control_points[1]],
                                )
                            ),
                        )
                    )
                else:
                    edit_actions.append(
                        MakeCurve(
                            type="arc",
                            control_points=list(
                                map(
                                    lambda x: (
                                        x[0] - render_config.grid_size // 2,
                                        x[1] - render_config.grid_size // 2,
                                    ),
                                    control_points,
                                )
                            ),
                        )
                    )
        elif a[0] == "mv_pt":
            edit_actions.append(
                MovePoint(
                    point=(
                        a[1] - render_config.grid_size // 2,
                        a[2] - render_config.grid_size // 2,
                    ),
                    new_point=(
                        a[3] - render_config.grid_size // 2,
                        a[4] - render_config.grid_size // 2,
                    ),
                )
            )
        elif a[0] == "mv_curve":
            curve, offset = a[1:]
            curve_type, *control_points = curve
            if curve_type == "line":
                edit_actions.append(
                    MoveCurve(
                        type=curve_type,
                        control_points=list(
                            map(
                                lambda x: (
                                    x[0] - render_config.grid_size // 2,
                                    x[1] - render_config.grid_size // 2,
                                ),
                                control_points,
                            )
                        ),
                        offset=offset,
                    )
                )
            else:
                if control_points[0] == control_points[-1]:
                    edit_actions.append(
                        MoveCurve(
                            type="circle",
                            control_points=list(
                                map(
                                    lambda x: (
                                        x[0] - render_config.grid_size // 2,
                                        x[1] - render_config.grid_size // 2,
                                    ),
                                    [control_points[0], control_points[1]],
                                )
                            ),
                            offset=offset,
                        )
                    )
                else:
                    edit_actions.append(
                        MoveCurve(
                            type="arc",
                            control_points=list(
                                map(
                                    lambda x: (
                                        x[0] - render_config.grid_size // 2,
                                        x[1] - render_config.grid_size // 2,
                                    ),
                                    control_points,
                                )
                            ),
                            offset=offset,
                        )
                    )
        elif a[0] == "rm_curve":
            curve_type, *control_points = a[1]
            if curve_type == "line":
                edit_actions.append(
                    RemoveCurve(
                        type=curve_type,
                        control_points=list(
                            map(
                                lambda x: (
                                    x[0] - render_config.grid_size // 2,
                                    x[1] - render_config.grid_size // 2,
                                ),
                                control_points,
                            )
                        ),
                    )
                )
            else:
                if control_points[0] == control_points[-1]:
                    edit_actions.append(
                        RemoveCurve(
                            type="circle",
                            control_points=list(
                                map(
                                    lambda x: (
                                        x[0] - render_config.grid_size // 2,
                                        x[1] - render_config.grid_size // 2,
                                    ),
                                    [control_points[0], control_points[1]],
                                )
                            ),
                        )
                    )
                else:
                    edit_actions.append(
                        RemoveCurve(
                            type="arc",
                            control_points=list(
                                map(
                                    lambda x: (
                                        x[0] - render_config.grid_size // 2,
                                        x[1] - render_config.grid_size // 2,
                                    ),
                                    control_points,
                                )
                            ),
                        )
                    )

    # get the extraneous curves produced by actions that aren't there in the design format of the execution
    extraneous_curves = get_extraneous_curves(
        execution, EditExecution.execute(context, edit_actions).design
    )

    # filter out actions that create extraneous curves
    filtered_actions = filter_actions(edit_actions, extraneous_curves)

    return filtered_actions


def executeActions(actions):
    """
    Executes a list of ACTIONS and returns the geometries (points and curves) that are created/ modified by those actions.

    Note: there are slight discrepencies between this and the JS executing function.
    """

    pt_id_ctr = 1
    points = {}
    curves = []
    cur_geometries = set()

    def add_geometry(geometry):
        geometry_str = str(geometry)

        if geometry_str in cur_geometries:
            return False

        else:
            cur_geometries.add(geometry_str)
            if geometry["name"] == "pt":
                nonlocal pt_id_ctr
                pt_id = "pt_" + str(pt_id_ctr)
                points[pt_id] = geometry
                pt_id_ctr += 1
            else:
                curves.append(geometry)
            return True

    def update_geometry_rep(geometry_old, geometry_new):

        geometry_str_old = str(geometry_old)
        if geometry_str_old not in cur_geometries:
            return False
        else:
            cur_geometries.remove(geometry_str_old)
            cur_geometries.add(str(geometry_new))
            return True

    def get_line_repr_from_action(line_content):
        pt1_coord_i, pt1_coord_j = line_content[1]
        pt2_coord_i, pt2_coord_j = line_content[2]

        pt1_id = next(
            (
                pt_id
                for pt_id, pt_obj in points.items()
                if pt_obj["i"] == pt1_coord_i and pt_obj["j"] == pt1_coord_j
            ),
            None,
        )
        pt2_id = next(
            (
                pt_id
                for pt_id, pt_obj in points.items()
                if pt_obj["i"] == pt2_coord_i and pt_obj["j"] == pt2_coord_j
            ),
            None,
        )

        if isinstance(curve, Line):
            normalized_curve = Line(control_points=normalized_control_points)
        elif isinstance(curve, Arc):
            normalized_curve = Arc(control_points=normalized_control_points)
        elif isinstance(curve, Circle):
            normalized_curve = Circle(control_points=normalized_control_points)

    def get_arc_repr_from_action(arc_content):
        curve_name = arc_content[0]
        pt1_coord_i, pt1_coord_j = arc_content[1]
        pt2_coord_i, pt2_coord_j = arc_content[2]
        pt3_coord_i, pt3_coord_j = arc_content[3]

    return Design(curves=normalized_curves)
