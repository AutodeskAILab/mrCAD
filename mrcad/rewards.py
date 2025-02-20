from mrcad.env import State
from mrcad.design import Design, Line, Arc, Circle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    euclidean_distances,
    jaccard_score,
    pairwise_distances,
    f1_score,
)
from scipy.optimize import linear_sum_assignment
import copy
import math


def chamfer_reward(state: State):
    return -state.current_design.chamfer_distance(state.target)


def design_distance(design1: Design, design2: Design):
    """
    Calculates the distance between two designs based on the chamfer distance between the two designs.
    """
    return design1.design_distance(design2)
