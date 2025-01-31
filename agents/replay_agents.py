from mrcad.action import Drawing, Instruction, Execution
from mrcad.design import Design
from mrcad.agents import AbstractDesignerAgent, AbstractMakerAgent
from mrcad.action import Action
from mrcad.env_utils import Role


class ReplayDesigner(AbstractDesignerAgent):
    def __init__(self, turns: list[tuple[str, Drawing]]):
        self.turns = turns
        self.turn_idx = 0

    def act(self, target: Design, conversation_history: list[tuple[Design, Action]]):
        text, drawing = self.turns[self.turn_idx]
        self.turn_idx += 1
        return Instruction(
            text=text if isinstance(text, str) else None,
            drawing=Drawing(drawing=drawing),
        )


class ReplayMaker(AbstractMakerAgent):
    def __init__(self, turns: list[Design]):
        self.turns = turns
        self.turn_idx = 0

    def act(self, conversation_history: list[tuple[Design, Action]]):
        action = self.turns[self.turn_idx]
        self.turn_idx += 1
        return Execution(design=action)
