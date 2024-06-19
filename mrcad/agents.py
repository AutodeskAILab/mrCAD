from mrcad.env import DesignerObservation, MakerObservation
from mrcad.action import Drawing
from mrcad.design import Design


class AbstractDesignerAgent:
    def act(self, observation: DesignerObservation) -> tuple[str, Drawing]:
        raise NotImplementedError


class AbstractMakerAgent:
    def act(self, observation: MakerObservation) -> Design:
        raise NotImplementedError
