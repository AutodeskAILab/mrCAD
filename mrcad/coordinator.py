from typing import List, Tuple
from copy import deepcopy
from mrcad.agents import AbstractDesignerAgent, AbstractMakerAgent
from mrcad.env import mrCADEnvironment, State
from mrcad.design import Design
from mrcad.action import Action, Drawing, Instruction, Execution
from mrcad.env_utils import Role
from mrcad.rewards import design_distance


class SynchronousCoordinator:
    def __init__(
        self, target: Design, designer: AbstractDesignerAgent, maker: AbstractMakerAgent
    ):
        self.env = mrCADEnvironment(
            State(target=target),
            {
                Role.DESIGNER: lambda x: None,
                Role.MAKER: design_distance,
            },
        )
        self.designer = designer
        self.maker = maker

    def play(self):
        done, truncate = False, False
        observation = self.env.reset()
        rewards = list()
        while not done and not truncate:
            next_turn = self.env.state.get_next_turn_role()
            if next_turn == Role.DESIGNER:
                action = self.designer.act(
                    self.env.state.target, self.env.state.conversation_history
                )
            elif next_turn == Role.MAKER:
                action = self.maker.act(self.env.state.conversation_history)

            state, reward, done, truncate = self.env.step(action)
            rewards.append(reward)
            if done or truncate:
                break

        return state, rewards


class MakerEvaluationCoordinator:
    def __init__(
        self,
        target: Design,
        maker: AbstractMakerAgent,
        conversation_history: List[Tuple[Design, Action]],
    ):
        self.maker = maker
        self.target = target
        self.conversation_history = conversation_history

    def play(self):
        actions = list()
        turn_idx = 0
        while turn_idx < len(self.conversation_history):
            if self.conversation_history[turn_idx][1].role == Role.DESIGNER:
                turn_idx += 1
                continue

            action = self.maker.act(self.conversation_history[:turn_idx])
            actions.append(action)
            turn_idx += 1

        return actions
