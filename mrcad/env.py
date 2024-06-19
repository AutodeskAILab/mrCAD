from dataclasses import dataclass
from mrcad.design import Design
from mrcad.env_utils import Role, OutOfTurnError
from mrcad.action import Action, Drawing


@dataclass
class DesignerObservation:
    history: tuple[tuple["DesignerObservation", Action]]
    target: Design
    current_design: Design
    turn: Role


@dataclass
class MakerObservation:
    history: tuple[tuple["MakerObservation", Action]]
    current_design: Design
    instruction: tuple[str, Drawing]
    turn: Role


@dataclass
class State:
    history: tuple[tuple["State", Action]]
    target: Design
    current_design: Design
    turn: Role

    def observe(self, role: Role):
        if role == Role.DESIGNER:
            return DesignerObservation(
                history=tuple((s.observe(Role.DESIGNER), a) for s, a in self.history),
                target=self.target,
                current_design=self.current_design,
                turn=self.turn,
            )
        elif role == Role.MAKER:
            instruction = None
            if len(self.history) > 0:
                last_turn = self.history[-1]
                if last_turn[0].turn == Role.DESIGNER:
                    instruction = None
                elif last_turn[0].turn == Role.MAKER:
                    instruction = last_turn[1].instruction
            return MakerObservation(
                history=tuple((s.observe(Role.MAKER), a) for s, a in self.history),
                current_design=self.current_design,
                turn=self.turn,
                instruction=instruction,
            )
        else:
            raise ValueError(f"Invalid role: {role}")

    def update(self, action: Action):
        if self.turn != action.role:
            raise OutOfTurnError

        if action.role == Role.DESIGNER:
            assert action.design is None, "Designer cannot execute a design"
            return State(
                history=(*self.history, (self, action)),
                target=self.target,
                current_design=self.current_design,
                turn=Role.MAKER,
            )

        if action.role == Role.MAKER:
            assert action.instruction is None, "Maker cannot provide an instruction"
            return State(
                history=(*self.history, (self, action)),
                target=self.target,
                current_design=action.design,
                turn=Role.DESIGNER,
            )

    @classmethod
    def initial(cls, target: Design):
        return cls(
            history=(),
            target=target,
            current_design=Design(tuple()),
            turn=Role.DESIGNER,
        )


@dataclass
class mrCADEnvironment:
    state: State
    reward_fns: dict[Role, callable]
    max_rounds: int = 3

    def reset(self):
        self.state = State.initial(self.state.target)
        return {role: self.state.observe(role) for role in iter(Role)}

    def step(self, action: Action):
        self.state = self.state.update(action)
        return (
            {role: self.state.observe(role) for role in iter(Role)},
            {role: self.reward_fns[role](self.state) for role in iter(Role)},
            False,
            len(self.state.history) >= 2 * self.max_rounds,
            None,  # infos are not used in this environment but kept for compatibility
        )
