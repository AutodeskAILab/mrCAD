from typing import Tuple, List, Optional, Literal, Union, Annotated
from dataclasses import dataclass
from mrcad import Design, Line, Arc, Circle, Execution
from pydantic import BaseModel, Field
import numpy as np


class DeletePoint(BaseModel):
    edit_type: Literal["delete_point"] = "delete_point"
    point: Tuple[float, float]

    def __call__(self, design: Design):
        new_curves = []
        for curve in design.curves:
            if self.point in curve.control_points:
                continue
            new_curves.append(curve)
        return Design(curves=new_curves)


class MovePoint(BaseModel):
    edit_type: Literal["move_point"] = "move_point"
    point: Tuple[float, float]
    new_point: Tuple[float, float]

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


class MakeCurve(BaseModel):
    edit_type: Literal["make_curve"] = "make_curve"
    type: Literal["line", "arc", "circle"]
    control_points: Tuple[Tuple[float, float], ...]

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


class MoveCurve(BaseModel):
    edit_type: Literal["move_curve"] = "move_curve"
    type: Literal["line", "arc", "circle"]
    control_points: Tuple[Tuple[float, float], ...]
    offset: Tuple[float, float]

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


class RemoveCurve(BaseModel):
    edit_type: Literal["remove_curve"] = "remove_curve"
    type: Literal["line", "arc", "circle"]
    control_points: Tuple[Tuple[float, float], ...]

    def __call__(self, design: Design):
        new_curves = []
        for curve in design.curves:
            if self.type == curve.type and curve.control_points == self.control_points:
                continue
            new_curves.append(curve)
        return Design(curves=new_curves)


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
