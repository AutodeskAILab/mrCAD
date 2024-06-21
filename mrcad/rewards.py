from mrcad.env import State


def chamfer_reward(state: State):
    return -state.current_design.chamfer_distance(state.target)
