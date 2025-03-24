from typing import List, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field
from mrcad.design import Design
from mrcad.env_utils import Role, OutOfTurnError
from mrcad.action import Action, Execution


class State(BaseModel):
    target: Design
    conversation_history: List[Tuple[Design, Action]] = Field(default_factory=list)

    def get_next_turn_role(self):
        if len(self.conversation_history) == 0:
            return Role.DESIGNER
        elif self.conversation_history[-1][1].role == Role.DESIGNER:
            return Role.MAKER
        elif self.conversation_history[-1][1].role == Role.MAKER:
            return Role.DESIGNER
        else:
            raise ValueError("Invalid conversation history")

    def update(self, action: Action):
        if action.role != self.get_next_turn_role():
            raise OutOfTurnError()

        if action.role == Role.DESIGNER:
            current_design = (
                self.conversation_history[-1][0]
                if len(self.conversation_history) > 0
                else Design(curves=[])
            )
            return State(
                target=self.target,
                conversation_history=[
                    *self.conversation_history,
                    (current_design, action),
                ],
            )

        if action.role == Role.MAKER:
            return State(
                target=self.target,
                conversation_history=[
                    *self.conversation_history,
                    (action.design, Execution()),
                ],
            )


@dataclass
class mrCADEnvironment:
    state: State
    steps = 0
    reward_fns: dict[Role, callable]
    max_steps: int = 6

    def reset(self):
        self.state = State(target=self.state.target)
        return self.state

    def step(self, action: Action):
        self.state = self.state.update(action)
        self.steps += 1
        return (
            self.state,
            {role: self.reward_fns[role](self.state) for role in iter(Role)},
            False,
            self.steps >= self.max_steps,
        )
