from typing import Tuple, List, Optional
from dataclasses import dataclass
from mrcad.design import Design, Line, Arc, Circle


@dataclass
class EditingAction:
    def __call__(self, design: Design):
        raise NotImplementedError

    def to_json(self):
        raise NotImplementedError

    @classmethod
    def from_json(cls, action_json):
        action_type = action_json["action"]
        if action_type == "add_line":
            return AddLine(action_json["control_points"])
        elif action_type == "add_arc":
            return AddArc(action_json["control_points"])
        elif action_type == "add_circle":
            return AddCircle(action_json["control_points"])
        elif action_type == "remove_line":
            return RemoveLine(action_json["control_points"])
        elif action_type == "remove_arc":
            return RemoveArc(action_json["control_points"])
        elif action_type == "remove_circle":
            return RemoveCircle(action_json["control_points"])
        elif action_type == "delete_point":
            return DeletePoint(action_json["point"])
        elif action_type == "move_point":
            return MovePoint(action_json["point"], action_json["new_point"])
        else:
            raise ValueError(f"Unknown action type: {action_type}")


@dataclass
class DeletePoint(EditingAction):
    point: Tuple

    def __call__(self, design: Design):
        new_curves = []
        for curve in design.curves:
            if self.point in curve.control_points:
                continue
            new_curves.append(curve)
        return Design(tuple(new_curves))

    def to_json(self):
        return {"action": "delete_point", "point": self.point}


@dataclass
class MovePoint(EditingAction):
    point: Tuple
    new_point: Tuple

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
                        new_curves.append(Line(tuple(new_control_points)))
                elif isinstance(curve, Arc):
                    if new_control_points[0] == new_control_points[2]:
                        new_curves.append(Circle(tuple(new_control_points[:2])))
                    elif (
                        new_control_points[0] == new_control_points[1]
                        or new_control_points[1] == new_control_points[2]
                    ):
                        continue
                    else:
                        new_curves.append(Arc(tuple(new_control_points)))
                elif isinstance(curve, Circle):
                    if new_control_points[0] == new_control_points[1]:
                        continue
                    new_curves.append(Circle(tuple(new_control_points)))
            else:
                new_curves.append(curve)
        return Design(tuple(new_curves))

    def to_json(self):
        return {
            "action": "move_point",
            "point": self.point,
            "new_point": self.new_point,
        }


@dataclass
class AddLine(EditingAction):
    control_points: Tuple[Tuple[float, float], Tuple[float, float]]

    def __call__(self, design: Design):
        if self.control_points[0] == self.control_points[1]:
            return design
        return Design(tuple([*design.curves, Line(self.control_points)]))

    def to_json(self):
        return {"action": "add_line", "control_points": self.control_points}


@dataclass
class AddArc(EditingAction):
    control_points: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]

    def __call__(self, design: Design):
        if self.control_points[0] == self.control_points[2]:
            return Design(tuple([*design.curves, Circle(self.control_points[:2])]))
        elif (
            self.control_points[0] == self.control_points[1]
            or self.control_points[1] == self.control_points[2]
        ):
            return design
        return Design(tuple([*design.curves, Arc(self.control_points)]))

    def to_json(self):
        return {"action": "add_arc", "control_points": self.control_points}


@dataclass
class AddCircle(EditingAction):
    control_points: Tuple[Tuple[float, float], Tuple[float, float]]

    def __call__(self, design: Design):
        if self.control_points[0] == self.control_points[1]:
            return design
        return Design(tuple([*design.curves, Circle(self.control_points)]))

    def to_json(self):
        return {"action": "add_circle", "control_points": self.control_points}


@dataclass
class RemoveLine(EditingAction):
    control_points: Tuple[Tuple[float, float], Tuple[float, float]]

    def __call__(self, design: Design):
        line_to_remove = Line(self.control_points)
        new_curves = []
        for curve in design.curves:
            if curve == line_to_remove:
                continue
            new_curves.append(curve)
        return Design(tuple(new_curves))

    def to_json(self):
        return {"action": "remove_line", "control_points": self.control_points}


@dataclass
class RemoveArc(EditingAction):
    control_points: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]

    def __call__(self, design: Design):
        arc_to_remove = Arc(self.control_points)
        new_curves = []
        for curve in design.curves:
            if curve == arc_to_remove:
                continue
            new_curves.append(curve)
        return Design(tuple(new_curves))

    def to_json(self):
        return {"action": "remove_arc", "control_points": self.control_points}


@dataclass
class RemoveCircle(EditingAction):
    control_points: Tuple[Tuple[float, float], Tuple[float, float]]

    def __call__(self, design: Design):
        circle_to_remove = Circle(self.control_points)
        new_curves = []
        for curve in design.curves:
            if curve == circle_to_remove:
                continue
            new_curves.append(curve)
        return Design(tuple(new_curves))

    def to_json(self):
        return {"action": "remove_circle", "control_points": self.control_points}
