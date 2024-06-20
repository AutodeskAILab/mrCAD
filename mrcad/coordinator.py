from copy import deepcopy
from mrcad.agents import AbstractDesignerAgent, AbstractMakerAgent
from mrcad.env import mrCADEnvironment, State
from mrcad.design import Design
from mrcad.env_utils import Role
from mrcad.rewards import chamfer_reward


class SynchronousCoordinator:
    def __init__(
        self, target: Design, designer: AbstractDesignerAgent, maker: AbstractMakerAgent
    ):
        self.env = mrCADEnvironment(
            State.initial(target),
            {Role.DESIGNER: chamfer_reward, Role.MAKER: chamfer_reward},
        )
        self.designer = designer
        self.maker = maker

    def play(self):
        done, truncate = False, False
        observation = self.env.reset()
        trajectory = []
        while not done and not truncate:
            current_state = deepcopy(self.env.state)
            if self.env.state.turn == Role.DESIGNER:
                action = self.designer.act(observation[Role.DESIGNER])
            elif self.env.state.turn == Role.MAKER:
                action = self.maker.act(observation[Role.MAKER])

            observation, rewards, done, truncate, _ = self.env.step(action)
            trajectory.append((current_state, action, rewards, self.env.state))
            if done or truncate:
                break

        return trajectory
