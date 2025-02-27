from typing import Tuple, List, Optional, Literal, Union, Annotated
from dataclasses import dataclass
from mrcad import Design, Line, Arc, Circle, Execution
from pydantic import BaseModel, Field
import numpy as np


class DeletePoint(BaseModel):
    """
    Delete a point from the design. Every curve that has a control point at the given point will be removed.
    """

    edit_type: Literal["delete_point"] = "delete_point"
    point: Tuple[float, float] = Field(..., description="The point to delete.")

    def __call__(self, design: Design):
        new_curves = []
        for curve in design.curves:
            if self.point in curve.control_points:
                continue
            new_curves.append(curve)
        return Design(curves=new_curves)

    def make_tool_call(self):
        return {"name": "delete_point", "arguments": {"point": self.point}}


class MovePoint(BaseModel):
    """
    Move a point to a new location. Every curve that has a control point at the given point will have the control point moved to the new location.
    """

    edit_type: Literal["move_point"] = "move_point"
    point: Tuple[float, float] = Field(..., description="The point to move.")
    new_point: Tuple[float, float] = Field(
        ..., description="The new location of the point."
    )

    def __call__(self, design: Design):
        new_curves = []
        for curve in design.curves:
            if self.point in curve.control_points:
                new_control_points = []
                for control_point in curve.control_points:
                    if control_point == self.point:
                        new_control_points.append(self.new_point)
                    else:
                        new_control_points.append(control_point)
                if isinstance(curve, Line):
                    if new_control_points[0] == new_control_points[1]:
                        continue
                    else:
                        new_curves.append(Line(control_points=new_control_points))
                elif isinstance(curve, Arc):
                    if new_control_points[0] == new_control_points[2]:
                        new_curves.append(Circle(control_points=new_control_points[:2]))
                    elif (
                        new_control_points[0] == new_control_points[1]
                        or new_control_points[1] == new_control_points[2]
                    ):
                        continue
                    else:
                        new_curves.append(Arc(control_points=new_control_points))
                elif isinstance(curve, Circle):
                    if new_control_points[0] == new_control_points[1]:
                        continue
                    new_curves.append(Circle(control_points=new_control_points))
            else:
                new_curves.append(curve)
        return Design(curves=new_curves)

    def make_tool_call(self):
        return {
            "name": "move_point",
            "arguments": {"point": self.point, "new_point": self.new_point},
        }


class MakeCurve(BaseModel):
    """
    Add a new curve to the design given the type of the curve and its control points.
    """

    edit_type: Literal["make_curve"] = "make_curve"
    type: Literal["line", "arc", "circle"] = Field(
        ..., description="The type of curve to make."
    )
    control_points: Tuple[Tuple[float, float], ...] = Field(
        ...,
        description="The control points of the curve. Each control point is a list with a pair of floating point numbers that represent the coordinates of the point on the canvas. A line contains two points, an arc contains three points, and a circle contains two points.",
    )

    def __call__(self, design: Design):
        if self.type == "line":
            return Design(
                curves=[*design.curves, Line(control_points=self.control_points)]
            )
        elif self.type == "arc":
            return Design(
                curves=[*design.curves, Arc(control_points=self.control_points)]
            )
        elif self.type == "circle":
            return Design(
                curves=[*design.curves, Circle(control_points=self.control_points)]
            )

    def make_tool_call(self):
        return {
            "name": "make_curve",
            "arguments": {"type": self.type, "control_points": self.control_points},
        }


class MoveCurve(BaseModel):
    """
    Move a curve to a new location. The curve to be moved is identified by its type and control points. The curve will be moved by the given offset.
    """

    edit_type: Literal["move_curve"] = "move_curve"
    type: Literal["line", "arc", "circle"] = Field(
        ..., description="The type of the curve to move."
    )
    control_points: Tuple[Tuple[float, float], ...] = Field(
        ...,
        description="The control points of the curve to move. Each control point is a list with a pair of floating point numbers that represent the coordinates of the point on the canvas.",
    )
    offset: Tuple[float, float] = Field(
        ...,
        description="The offset to move the curve. The offset is a pair of floating point numbers that represent how much the curve has to be moved in the x and y directions respectively.",
    )

    def __call__(self, design: Design):
        new_curves = []
        for i, curve in enumerate(design.curves):
            if self.type == curve.type and curve.control_points == self.control_points:
                new_control_points = tuple(
                    (np.array(control_point) + np.array(self.offset)).tolist()
                    for control_point in curve.control_points
                )
                if self.type == "line":
                    new_curves.append(Line(control_points=new_control_points))
                elif self.type == "arc":
                    new_curves.append(Arc(control_points=new_control_points))
                elif self.type == "circle":
                    new_curves.append(Circle(control_points=new_control_points))
            else:
                new_curves.append(curve)
        return Design(curves=new_curves)

    def make_tool_call(self):
        return {
            "name": "move_curve",
            "arguments": {
                "type": self.type,
                "control_points": self.control_points,
                "offset": self.offset,
            },
        }


class RemoveCurve(BaseModel):
    """
    Remove a curve from the design. The curve to be removed is identified by its type and control points.
    """

    edit_type: Literal["remove_curve"] = "remove_curve"
    type: Literal["line", "arc", "circle"] = Field(
        ..., description="The type of the curve to remove."
    )
    control_points: Tuple[Tuple[float, float], ...] = Field(
        ..., description="The control points of the curve to remove."
    )

    def __call__(self, design: Design):
        new_curves = []
        for curve in design.curves:
            if self.type == curve.type and curve.control_points == self.control_points:
                continue
            new_curves.append(curve)
        return Design(curves=new_curves)

    def make_tool_call(self):
        return {
            "name": "remove_curve",
            "arguments": {"type": self.type, "control_points": self.control_points},
        }


Edit = Annotated[
    Union[DeletePoint, MovePoint, MakeCurve, MoveCurve, RemoveCurve],
    Field(discriminator="edit_type"),
]


class EditExecution(Execution):
    edits: Tuple[Edit, ...]

    @classmethod
    def execute(cls, design: Design, edits: List[Edit]):
        edited_design = Design(curves=design.curves)
        for edit in edits:
            edited_design = edit(edited_design)
        return cls(design=edited_design, edits=edits)
