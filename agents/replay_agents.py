from typing import List, Tuple
from mrcad import Design, Drawing, Instruction, Execution, Action
from mrcad.agents import AbstractDesignerAgent, AbstractMakerAgent
from mrcad.env_utils import Role


class ReplayDesigner(AbstractDesignerAgent):
    def __init__(self, turns: List[Tuple[Design, Action]]):
        self.turns = turns
        self.turn_idx = 0

    def act(
        self, target: Design, conversation_history: List[Tuple[Design, Action]]
    ) -> Action:
        context, instruction = self.turns[self.turn_idx]
        self.turn_idx += 1
        return instruction


class ReplayMaker(AbstractMakerAgent):
    def __init__(self, turns: List[Tuple[Design, Action]]):
        self.turns = turns
        self.turn_idx = 0

    def act(self, conversation_history: List[Tuple[Design, Action]]) -> Action:
        context, execution = self.turns[self.turn_idx]
        self.turn_idx += 1
        return execution
