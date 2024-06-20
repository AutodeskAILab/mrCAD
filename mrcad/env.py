from dataclasses import dataclass
from mrcad.design import Design
from mrcad.env_utils import Role, OutOfTurnError
from mrcad.action import Action, Drawing


@dataclass
class DesignerObservation:
    target: Design
    current_design: Design
    turn: Role


@dataclass
class MakerObservation:
    current_design: Design
    instruction: tuple[str, Drawing]
    turn: Role


@dataclass
class State:
    target: Design
    current_design: Design
    instruction: tuple[str, Drawing]
    turn: Role

    def observe(self, role: Role):
        if role == Role.DESIGNER:
            return DesignerObservation(
                target=self.target,
                current_design=self.current_design,
                turn=self.turn,
            )
        elif role == Role.MAKER:
            return MakerObservation(
                current_design=self.current_design,
                turn=self.turn,
                instruction=self.instruction,
            )
        else:
            raise ValueError(f"Invalid role: {role}")

    def update(self, action: Action):
        if self.turn != action.role:
            raise OutOfTurnError

        if action.role == Role.DESIGNER:
            assert action.design is None, "Designer cannot execute a design"
            return State(
                target=self.target,
                current_design=self.current_design,
                instruction=action.instruction,
                turn=Role.MAKER,
            )

        if action.role == Role.MAKER:
            assert action.instruction is None, "Maker cannot provide an instruction"
            return State(
                target=self.target,
                current_design=action.design,
                instruction=None,
                turn=Role.DESIGNER,
            )

    @classmethod
    def initial(cls, target: Design):
        return cls(
            target=target,
            current_design=Design(tuple()),
            instruction=None,
            turn=Role.DESIGNER,
        )


@dataclass
class mrCADEnvironment:
    state: State
    steps = 0
    reward_fns: dict[Role, callable]
    max_steps: int = 6

    def reset(self):
        self.state = State.initial(self.state.target)
        return {role: self.state.observe(role) for role in iter(Role)}

    def step(self, action: Action):
        self.state = self.state.update(action)
        self.steps += 1
        return (
            {role: self.state.observe(role) for role in iter(Role)},
            {role: self.reward_fns[role](self.state) for role in iter(Role)},
            False,
            self.steps >= self.max_steps,
            None,  # infos are not used in this environment but kept for compatibility
        )
