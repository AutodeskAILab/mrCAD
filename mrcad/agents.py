from typing import List, Tuple
from mrcad.action import Action
from mrcad.design import Design


class AbstractDesignerAgent:
    def act(
        self, target: Design, conversation_history: List[Tuple[Design, Action]]
    ) -> Action:
        raise NotImplementedError


class AbstractMakerAgent:
    def act(self, conversation_history: List[Tuple[Design, Action]]) -> Design:
        raise NotImplementedError
