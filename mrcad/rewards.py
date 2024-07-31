from mrcad.env import State
from mrcad.design import Design, Line, Arc, Circle


def chamfer_reward(state: State):
    return -state.current_design.chamfer_distance(state.target)


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


#### METRICS


## Program metrics


def entity_similarity_with_circles(ent1, ent2, W=41):
    """
    Fairly naive measure of similarity between two geometric enitites (points, lines, arcs, and circles).

    Opertates on compressed representations of geometries i.e.:
        line = ((x,y),(z,w))
        arc = ((x,y),(z,w),(a,b))

    Grid assumed to be 20 squares in all directions (plus axes), so 41.
    """

    def pt_distance(x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def line_sim(l1, l2):
        x1, y1, x2, y2 = l1[0][0], l1[0][1], l1[1][0], l1[1][1]
        x3, y3, x4, y4 = l2[0][0], l2[0][1], l2[1][0], l2[1][1]
        pt1_dist = pt_distance(x1, y1, x3, y3)
        pt2_dist = pt_distance(x2, y2, x4, y4)
        avg_dist = (pt1_dist + pt2_dist) / 2.0
        # do an exp decay for similiarity
        return math.exp(-avg_dist / W)

    def arc_sim(a1, a2):
        x1, y1, x2, y2, x3, y3 = (
            a1[0][0],
            a1[0][1],
            a1[1][0],
            a1[1][1],
            a1[2][0],
            a1[2][1],
        )
        x4, y4, x5, y5, x6, y6 = (
            a2[0][0],
            a2[0][1],
            a2[1][0],
            a2[1][1],
            a2[2][0],
            a2[2][1],
        )
        pt1_dist = pt_distance(x1, y1, x4, y4)
        pt2_dist = pt_distance(x2, y2, x5, y5)
        pt3_dist = pt_distance(x3, y3, x6, y6)
        avg_dist = (pt1_dist + pt2_dist + pt3_dist) / 3.0
        # do an exp decay for similiarity
        return math.exp(-avg_dist / W)

    def circ_sim(c1, c2):
        x1, y1, x2, y2, x3, y3 = (
            c1[0][0],
            c1[0][1],
            c1[1][0],
            c1[1][1],
            c1[2][0],
            c1[2][1],
        )
        x4, y4, x5, y5, x6, y6 = (
            c2[0][0],
            c2[0][1],
            c2[1][0],
            c2[1][1],
            c2[2][0],
            c2[2][1],
        )

        c1_center_x = (x2 - x1) / 2 + x1
        c1_center_y = (y2 - y1) / 2 + y1
        c1_radius = math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))

        c2_center_x = (x5 - x4) / 2 + x1
        c2_center_y = (y5 - y4) / 2 + y1
        c2_radius = math.sqrt(((x5 - x4) ** 2) + ((y5 - y4) ** 2))

        center_dist = pt_distance(c2_center_x, c2_center_y, c1_center_x, c1_center_y)
        rad_dist = abs(c1_radius - c2_radius)

        # kind of arbitrary right now- just average location and radius distances
        circle_dist = (center_dist + 2 * rad_dist) / 2

        #         avg_dist = (pt1_dist + pt2_dist + pt3_dist)/3.0

        # do an exp decay for similiarity
        return math.exp(-circle_dist / W)

    # if they are different length, i.e. a line and an arc, return 0
    if len(ent1) != len(ent2):
        return 0.0
    # if they are both lines
    if len(ent1) == 2:
        # return the max with reversed drawing order in mind
        return max(line_sim(ent1, ent2), line_sim(ent1, (ent2[1], ent2[0])))
    # if they are both arcs, compare with first and last elements swapped
    if len(ent1) == 3:
        # if they are both circles
        if (ent1[0] == ent1[2]) and (ent2[0] == ent2[2]):
            return circ_sim(ent1, ent2)
        else:
            return max(arc_sim(ent1, ent2), arc_sim(ent1, (ent2[2], ent2[1], ent2[0])))


def calculate_similarity_matrix(cad1, cad2, similarity_function):

    cad = copy.deepcopy(cad1)
    cad_other = copy.deepcopy(cad2)

    len1, len2 = len(cad), len(cad_other)
    similarity_matrix = np.zeros((len1, len2))

    for i in range(len1):
        for j in range(len2):
            similarity_matrix[i, j] = similarity_function(cad[i], cad_other[j])

    return similarity_matrix


def optimal_sort(similarity_matrix, maximize=True):
    """
    Takes a distance matrix and returns a sorted version of the matrix and the optimal assignment.
    The sorting is based on the Hungarian algorithm (linear sum assignment) to either maximize
    or minimize the overall distance/similarity between the pairs.

    Parameters
    ----------
    distance_matrix: ndarray
        A 2D NumPy array representing the distance matrix where each element (i, j) is the distance
        or similarity between the i-th and j-th elements of two sequences.

    maximize: bool, optional
        A boolean flag that determines whether to maximize the mean similarity (if set to True)
        or minimize the mean distance (if set to False) between the pairs in the distance matrix.
        Default is True.

    Returns
    -------
    sorted_matrix: ndarray
        A 2D NumPy array that represents the distance matrix sorted according to the optimal assignment.
        This sorting ensures maximization or minimization of the total distance/similarity.

    optimal_assignment: ndarray
        A 1D NumPy array containing the indices of the sorted elements in the second dimension
        of the distance matrix. It represents the optimal pairing/ordering of the elements.
    """

    row_ind, col_ind = linear_sum_assignment(similarity_matrix, maximize=maximize)
    #     sorted_matrix = similarity_matrix[row_ind, col_ind]
    element_distances = similarity_matrix[row_ind, col_ind]

    return element_distances, col_ind


def orderless_distance(
    cad1, cad2, element_similarity_measure, cad_similarity_measure, truncate=False
):
    """
    Computes the distance matrix between two sets of geometries, finding best possible one-to-one mapping between stims.

    If truncate = True, take the average difference between all matched elements, otherwise treat unmatched elements as distance = 0.

    """
    similarity_matrix = calculate_similarity_matrix(
        cad1, cad2, element_similarity_measure
    )

    #     print('len c1', len(cad1))
    #     print('len c2', len(cad2))

    element_distances, col_ind = optimal_sort(similarity_matrix)

    if truncate:

        if len(element_distances) == 0:
            return 0
        else:
            return cad_similarity_measure(element_distances)

    else:
        # find number of unmatched elements
        unmatched_element_distances = np.zeros(
            max(len(cad1), len(cad2)) - len(element_distances)
        )

        # pad distance array with 0s
        element_distances_padded = np.concatenate(
            (element_distances, unmatched_element_distances)
        )

        # aggregate element distances
        return cad_similarity_measure(element_distances_padded)


def accuracy_mean_element_distance(cad1, cad2, truncate=False):
    """
    Calculates a distance between two sets of geometries by matching elements in each.

    We find the optimal mapping between entities and returns the mean distance between them.
    We count unmatched elements as zero.

    """
    return orderless_distance(
        cad1, cad2, entity_similarity_with_circles, np.mean, truncate=truncate
    )


def shape_loss(geoms_input, geoms_target):
    """
    Returns the total number of additional elements (of each type) required to match distribution of types in stimulus,
    expressed as a proportion of target geometries.
    When the number of changes exceeds the number of geometries in the target, this value is capped at 1.


    Returns: number in range [0, 1]

    """

    def element_counts(geoms):
        """
        How many elements of each type are there?
        """
        count = {"lines": 0, "arcs": 0, "circles": 0}

        for el in geoms:
            if len(el) == 2:
                count["lines"] = count["lines"] + 1
            elif el[0] == el[2]:
                count["circles"] = count["circles"] + 1
            else:
                count["arcs"] = count["arcs"] + 1
        return count

    counts_input = element_counts(geoms_input)
    counts_target = element_counts(geoms_target)

    #     print(counts_input, counts_target)

    # initially, just count number of changes required (adds or deletions)
    count_abs_diffs = {
        k: np.abs(counts_target[k] - counts_input[k]) for k in counts_input.keys()
    }

    changes_needed = sum(count_abs_diffs.values())

    # what proportion of the stimulus elements need changing?
    # if more than 100%, cap at 1
    # if all elements need changing (or more than double the number of elements exist) then cap loss at 1
    prop_elements_requiring_change = min(1, changes_needed / len(geoms_target))

    return prop_elements_requiring_change


def control_point_loss(geoms_input, geoms_target):
    # HERE: Explore without having 0 for non-matches
    return 1 - accuracy_mean_element_distance(geoms_input, geoms_target, truncate=True)


def geometry_loss(geoms_input, geoms_target):
    """
    geometric mean of shape loss and control point loss
    """

    # geometric mean (adapted so that 0 loss in either doesn't necessarily lead to 0 loss overall
    #     loss = 1 - np.sqrt((1 - control_point_loss(geoms_input, geoms_target)) * (1-shape_loss(geoms_input, geoms_target)))

    # arithmetic mean
    #     loss = (control_point_loss(geoms_input, geoms_target) + shape_loss(geoms_input, geoms_target))/2

    # harmonic mean
    loss = 1 - 2 / (
        1 / (1.00000001 - control_point_loss(geoms_input, geoms_target))
        + 1 / (1.00000001 - shape_loss(geoms_input, geoms_target))
    )

    return loss
