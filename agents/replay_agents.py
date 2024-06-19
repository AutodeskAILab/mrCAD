from mrcad.action import Drawing
from mrcad.design import Design
from mrcad.agents import AbstractDesignerAgent, AbstractMakerAgent
from mrcad.action import Action
from mrcad.env_utils import Role


class ReplayDesigner(AbstractDesignerAgent):
    def __init__(self, turns: list[tuple[str, Drawing]]):
        self.turns = turns
        self.turn_idx = 0

    def act(self, observation):
        action = self.turns[self.turn_idx]
        self.turn_idx += 1
        return Action(Role.DESIGNER, action, None)


class ReplayMaker(AbstractMakerAgent):
    def __init__(self, turns: list[Design]):
        self.turns = turns
        self.turn_idx = 0

    def act(self, observation):
        action = self.turns[self.turn_idx]
        self.turn_idx += 1
        return Action(Role.MAKER, None, action)
