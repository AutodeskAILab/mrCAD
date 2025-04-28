import json
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import (
    euclidean_distances,
    jaccard_score,
    pairwise_distances,
    f1_score,
)
from scipy.optimize import linear_sum_assignment
import math


# MAX_DISTANCE = 210
DEFAULT_MAX_DISTANCE = 140
# ================== loading the curves ...  ==================


# parse the curves into their respective types, in the original format
# (line, pt1, pt2)
# (circle, pt1, pt2)
# (arc, pt1, pt2, pt3)
def parse_curves(json_data):
    curves = []
    for curve in json_data["curves"]:
        if curve["name"] == "line":
            pt1_name, pt2_name = curve["pt1"], curve["pt2"]
            pt1_coord = (
                json_data["points"][pt1_name]["i"],
                json_data["points"][pt1_name]["j"],
            )
            pt2_coord = (
                json_data["points"][pt2_name]["i"],
                json_data["points"][pt2_name]["j"],
            )
            curves.append(("line", pt1_coord, pt2_coord))

        if curve["name"] == "arc_3pt":
            start, mid, end = curve["start"], curve["mid"], curve["end"]
            # this is a circle
            if start == end:
                start_coord = (
                    json_data["points"][start]["i"],
                    json_data["points"][start]["j"],
                )
                mid_coord = json_data["points"][mid]["i"], json_data["points"][mid]["j"]
                curves.append(("circle", start_coord, mid_coord))
            # this is an arc
            else:
                start_coord = (
                    json_data["points"][start]["i"],
                    json_data["points"][start]["j"],
                )
                mid_coord = json_data["points"][mid]["i"], json_data["points"][mid]["j"]
                end_coord = json_data["points"][end]["i"], json_data["points"][end]["j"]
                curves.append(("arc", start_coord, mid_coord, end_coord))

    return curves


# ================== logics about lines ==================


# canonicalize the line
# from (line, pt1, pt2)
# to (line, pt1, pt2)
# after some deliberation, we decided to keep the original format
def canonicalize_line(line):
    line_type, pt1, pt2 = line
    return (line_type, pt1, pt2)


# distance between two canonicalized lines
def canonicalized_line_dist(line1, line2):
    line1_type, line1_pt1, line1_pt2 = line1
    line2_type, line2_pt1, line2_pt2 = line2

    # get the 4 pairs of distances
    dist_pt1_pt1 = (
        (line1_pt1[0] - line2_pt1[0]) ** 2 + (line1_pt1[1] - line2_pt1[1]) ** 2
    ) ** 0.5
    dist_pt1_pt2 = (
        (line1_pt1[0] - line2_pt2[0]) ** 2 + (line1_pt1[1] - line2_pt2[1]) ** 2
    ) ** 0.5
    dist_pt2_pt1 = (
        (line1_pt2[0] - line2_pt1[0]) ** 2 + (line1_pt2[1] - line2_pt1[1]) ** 2
    ) ** 0.5
    dist_pt2_pt2 = (
        (line1_pt2[0] - line2_pt2[0]) ** 2 + (line1_pt2[1] - line2_pt2[1]) ** 2
    ) ** 0.5

    # one arrangement is pt1 to pt1, pt2 to pt2
    dist1 = dist_pt1_pt1 + dist_pt2_pt2
    # the other arrangement is pt1 to pt2, pt2 to pt1
    dist2 = dist_pt1_pt2 + dist_pt2_pt1

    # return the minimum of the two
    distance = min(dist1, dist2)

    if not math.isnan(distance):
        return distance
    else:
        # print (line1, line2)
        raise ValueError("distance is nan")


# ================== logics about circles ==================


# canonicalize the circle
# from (circle, pt1, pt2)
# to (circle, origin, radius)
def canonicalize_circle(circle):
    circle_type, pt1, pt2 = circle
    # origin is beween pt1 and pt2
    origin = ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)
    # radius is pt1 to origin
    radius = ((pt1[0] - origin[0]) ** 2 + (pt1[1] - origin[1]) ** 2) ** 0.5
    # print('origin', origin)
    # print('radius', radius)
    # print('origin', origin)
    # print('radius', radius)
    return (circle_type, origin, radius)


# distance between two canonicalized circles
def canonicalized_circle_dist(circle1, circle2):
    circle1_type, circle1_origin, circle1_radius = circle1
    circle2_type, circle2_origin, circle2_radius = circle2

    # get the distance between the origins
    dist_origin = (
        (circle1_origin[0] - circle2_origin[0]) ** 2
        + (circle1_origin[1] - circle2_origin[1]) ** 2
    ) ** 0.5
    diff_radius = abs(circle1_radius - circle2_radius)

    distance = dist_origin + (2 * diff_radius)

    # if distance is numerical return it, otherwise throw error and print the circles
    if not math.isnan(distance):
        return distance
    else:
        # print ('err', circle1, circle2)
        raise ValueError("distance is nan")


# ================== logic about arcs ==================


# center and radius of circle, bug free
def find_circle_center(x1, y1, x2, y2, x3, y3):

    # print('cc points', x1, y1, x2, y2, x3, y3)

    # If the first point coincides with the third point, it is a full circle
    if x1 == x3 and y1 == y3:
        # Return the midpoint of the first and second point
        ret = [(x1 + x2) / 2, (y1 + y2) / 2]
        print(ret)
        return ret

    x12 = x1 - x2
    x13 = x1 - x3

    y12 = y1 - y2
    y13 = y1 - y3

    y31 = y3 - y1
    y21 = y2 - y1

    x31 = x3 - x1
    x21 = x2 - x1

    # x1^2 - x3^2
    sx13 = x1**2 - x3**2

    # y1^2 - y3^2
    sy13 = y1**2 - y3**2

    sx21 = x2**2 - x1**2
    sy21 = y2**2 - y1**2

    f = (sx13 * x12 + sy13 * x12 + sx21 * x13 + sy21 * x13) / (
        2 * (y31 * x12 - y21 * x13)
    )
    g = (sx13 * y12 + sy13 * y12 + sx21 * y13 + sy21 * y13) / (
        2 * (x31 * y12 - x21 * y13)
    )

    # print('f,g:', f, g)

    c = -(x1**2) - y1**2 - 2 * g * x1 - 2 * f * y1

    # Equation of circle: x^2 + y^2 + 2*g*x + 2*f*y + c = 0
    # where the center is (h = -g, k = -f)
    h = -g
    k = -f

    return h, k


def canonicalize_arc(arc):
    arc_type, pt_start, pt_mid, pt_end = arc
    # somehow the radius returned by this function is not correct (  ? )
    center = find_circle_center(
        pt_start[0], pt_start[1], pt_mid[0], pt_mid[1], pt_end[0], pt_end[1]
    )

    # radius is distance from center to start point
    radius = ((pt_start[0] - center[0]) ** 2 + (pt_start[1] - center[1]) ** 2) ** 0.5

    # print('non-can arc', arc, 'center', center, 'radius', radius)
    # print('center', center)
    # print('radius', radius)

    # get the start angle
    start_angle = np.arctan2(pt_start[1] - center[1], pt_start[0] - center[0])
    # get the end angle
    end_angle = np.arctan2(pt_end[1] - center[1], pt_end[0] - center[0])
    # get the angle that the circle passes through the mid point
    mid_angle = np.arctan2(pt_mid[1] - center[1], pt_mid[0] - center[0])

    # print('start angle', start_angle)
    # print('end angle', end_angle)
    # print('mid angle', mid_angle)

    # expand the angles to be on the linear number line, e.g. smesmesmesme
    # doing it this way you will always find them in the right, increasing order, despite the discontinuity at 0/2pi
    expanded_list = [("s", start_angle), ("m", mid_angle), ("e", end_angle)]
    expanded_list.append(("s", start_angle + 2 * np.pi))
    expanded_list.append(("m", mid_angle + 2 * np.pi))
    expanded_list.append(("e", end_angle + 2 * np.pi))
    # sort this list by angles
    expanded_list.sort(key=lambda x: x[1])

    # print (f"[canon] expanded list: {expanded_list}")

    # convert it into a string of letters
    order_str = "".join([x[0] for x in expanded_list])
    assert "sme" in order_str or "ems" in order_str, "something is terribly wrong"

    # note the sequence:
    # 1. getting the angles in increasing order from the list
    # 2. centering the mid angle to be halfway between start and end (taking numerical average)
    # 3. capping the angles to be between 0 and 2pi
    # the steps 2 and 3 CANNOT be swapped, otherwise you lose the CW / CCW information
    # e.g. you'll have two arcs sharing start and end, but
    #      one half-moon to the left, one to the right, and you cannot tell them apart
    if "sme" in order_str:
        # grab the angles of sme from the expanded list
        sme_idx = order_str.index("sme")
        new_start_a, new_mid_a, new_end_a = [
            expanded_list[sme_idx + i][1] for i in range(3)
        ]
        # center the new_mid_a to be halfway between new_start_a and new_end_a
        new_mid_a = (new_start_a + new_end_a) / 2
        # cap the angles to be between 0 and 2pi
        new_start_a = new_start_a % (2 * np.pi)
        new_mid_a = new_mid_a % (2 * np.pi)
        new_end_a = new_end_a % (2 * np.pi)
    if "ems" in order_str:
        # grab the angles of sme from the expanded list
        ems_idx = order_str.index("ems")
        # swap the start with end
        new_start_a, new_mid_a, new_end_a = [
            expanded_list[ems_idx + i][1] for i in range(3)
        ]
        # center the new_mid_a to be halfway between new_start_a and new_end_a
        new_mid_a = (new_start_a + new_end_a) / 2
        # cap the angles to be between 0 and 2pi
        new_start_a = new_start_a % (2 * np.pi)
        new_mid_a = new_mid_a % (2 * np.pi)
        new_end_a = new_end_a % (2 * np.pi)

    # print (f"[canon-sme] start: {new_start_a}, mid: {new_mid_a}, end: {new_end_a}")

    # phew, finally
    # get the start, mid, end points
    new_start = (
        center[0] + radius * np.cos(new_start_a),
        center[1] + radius * np.sin(new_start_a),
    )
    new_mid = (
        center[0] + radius * np.cos(new_mid_a),
        center[1] + radius * np.sin(new_mid_a),
    )
    new_end = (
        center[0] + radius * np.cos(new_end_a),
        center[1] + radius * np.sin(new_end_a),
    )

    return (arc_type, center, new_start, new_mid, new_end)


# distance between two canonicalized arcs
def canonicalized_arc_dist(arc1, arc2):

    arc1_type, arc1_center, arc1_start, arc1_mid, arc1_end = arc1
    arc2_type, arc2_center, arc2_start, arc2_mid, arc2_end = arc2

    # move mid point to mid point of the two arcs, and get the distance
    dist_mid = (
        (arc1_mid[0] - arc2_mid[0]) ** 2 + (arc1_mid[1] - arc2_mid[1]) ** 2
    ) ** 0.5
    # option1: start1 to start2, end1 to end2
    dist_option1 = (
        (arc1_start[0] - arc2_start[0]) ** 2 + (arc1_start[1] - arc2_start[1]) ** 2
    ) ** 0.5 + (
        (arc1_end[0] - arc2_end[0]) ** 2 + (arc1_end[1] - arc2_end[1]) ** 2
    ) ** 0.5
    # option2: start1 to end2, end1 to start2
    dist_option2 = (
        (arc1_start[0] - arc2_end[0]) ** 2 + (arc1_start[1] - arc2_end[1]) ** 2
    ) ** 0.5 + (
        (arc1_end[0] - arc2_start[0]) ** 2 + (arc1_end[1] - arc2_start[1]) ** 2
    ) ** 0.5

    distance = dist_mid + min(dist_option1, dist_option2)

    # print('dist mid', dist_mid)
    # print('dist_option1', dist_option1)
    # print('dist_option2', dist_option2)
    # print('arc dist', arc1, arc2)
    # print(distance)

    return distance


# ================== canonicalize all curves ==================
def canonicalize_curves(curves):
    canonicalized_curves = []
    for curve in curves:
        curve_type, *curve_data = curve
        if curve_type == "line":
            canonicalized_curves.append(canonicalize_line(curve))
        elif curve_type == "circle":
            canonicalized_curves.append(canonicalize_circle(curve))
        elif curve_type == "arc":
            if is_collinear(curve_data[0], curve_data[1], curve_data[2]):
                # print('collinear arc converted to line')
                canonicalized_curves.append(["line", curve_data[0], curve_data[2]])
            else:
                canonicalized_curves.append(canonicalize_arc(curve))
    return canonicalized_curves


def is_collinear(pt1, pt2, pt3):
    """
    Check if three points are collinear
    """
    x1, y1 = pt1
    x2, y2 = pt2
    x3, y3 = pt3

    return (y1 - y2) * (x1 - x3) == (y1 - y3) * (x1 - x2)


def geom_distance(ent1, ent2, W=41, max_distance=DEFAULT_MAX_DISTANCE):
    """
    Compare two elements
    """

    # if they are different length, i.e. a line and an arc, return 0
    if ent1[0] != ent2[0]:
        return max_distance
    # if they are both lines
    elif ent1[0] == "line":
        # return the max with reversed drawing order in mind
        return min(
            canonicalized_line_dist(canonicalize_line(ent1), canonicalize_line(ent2)),
            canonicalized_line_dist(canonicalize_line(ent2), canonicalize_line(ent1)),
        )
    # if they are both arcs, compare with first and last elements swapped
    elif ent1[0] == "circle":
        # if they are both circles
        # print(ent1, ent2)
        return min(
            canonicalized_circle_dist(
                canonicalize_circle(ent1), canonicalize_circle(ent2)
            ),
            canonicalized_circle_dist(
                canonicalize_circle(ent2), canonicalize_circle(ent1)
            ),
        )
        # return min(canonicalized_circle_dist(ent1,ent2),
        #             canonicalized_circle_dist(ent1,ent2))

    elif ent1[0] == "arc":
        # print('trying arcs', ent1, ent2)

        # check if either is secretly a line/ are points are collinear
        if is_collinear(ent1[1], ent1[2], ent1[3]) or is_collinear(
            ent2[1], ent2[2], ent2[3]
        ):
            return max_distance

        canonicalized_arc_1 = canonicalize_arc(ent1)
        canonicalized_arc_2 = canonicalize_arc(ent2)

        # print(canonicalized_arc_1)
        # print(canonicalized_arc_2)
        distance = canonicalized_arc_dist(canonicalized_arc_1, canonicalized_arc_2)

        # print(distance)

        return distance


def calculate_distance_matrix(
    cad1, cad2, distance_function, max_distance=DEFAULT_MAX_DISTANCE
):

    cad = copy.deepcopy(cad1)
    cad_other = copy.deepcopy(cad2)

    len1, len2 = len(cad), len(cad_other)
    distance_matrix = np.zeros((len1, len2)) + max_distance

    for i in range(len1):
        for j in range(len2):
            distance = distance_function(
                cad[i], cad_other[j], max_distance=max_distance
            )
            # if not nan or None
            if distance is not None and not math.isnan(distance):
                distance_matrix[i, j] = distance
            else:
                # print (cad[i], cad_other[j])
                raise ValueError("distance is nan")

    return distance_matrix


def orderless_distance(
    cad1,
    cad2,
    element_distance_measure,
    cad_distance_measure,
    truncate=False,
    max_distance=DEFAULT_MAX_DISTANCE,
):
    """
    Computes the distance matrix between two sets of geometries, finding best possible one-to-one mapping between stims.

    If truncate = True, take the average difference between all matched elements, otherwise treat unmatched elements as distance = 0.

    """

    distance_matrix = calculate_distance_matrix(
        cad1, cad2, element_distance_measure, max_distance=max_distance
    )

    # print([c[0] for c in cad1])
    # print([c[0] for c in cad2])

    element_distances, col_ind = optimal_sort(distance_matrix)

    if truncate:

        if len(element_distances) == 0:
            return max_distance
        else:
            return cad_distance_measure(element_distances)

    else:
        # find number of unmatched elements
        unmatched_element_distances = (
            np.zeros(max(len(cad1), len(cad2)) - len(element_distances)) + max_distance
        )

        # pad distance array with max distance
        element_distances_padded = np.concatenate(
            (element_distances, unmatched_element_distances)
        )

        # aggregate element distances
        return cad_distance_measure(element_distances_padded)


def optimal_sort(distance_matrix, maximize=False):
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

    row_ind, col_ind = linear_sum_assignment(distance_matrix, maximize=maximize)
    #     sorted_matrix = similarity_matrix[row_ind, col_ind]
    element_distances = distance_matrix[row_ind, col_ind]

    return element_distances, col_ind


def accuracy_mean_element_distance(
    cad1, cad2, truncate=False, max_distance=DEFAULT_MAX_DISTANCE
):
    """
    Calculates a distance between two sets of geometries by matching elements in each.

    We find the optimal mapping between entities and returns the mean distance between them.
    We count unmatched elements as zero.

    """

    # print(cad1, cad2)

    return orderless_distance(
        cad1, cad2, geom_distance, np.mean, truncate=truncate, max_distance=max_distance
    )


def center_geometries(curves, y_offset, x_offset):
    """
    Centers the geometries in the scene by adjusting the x and y coordinates of the points.
    """
    centered_curves = []
    for curve in curves:
        curve_type, *points = curve
        centered_points = []
        for point in points:
            i, j = point
            centered_points.append((i + y_offset, j + x_offset))
        centered_curves.append((curve_type, *centered_points))
    return centered_curves


#### CHAMFER METRICS
N_CURVE_PTS = 10

# ================== logics about lines ==================

# # distance canonicalized line segment to a point
# def line_dist_to_pt(segment, target_pt):
#     # print (segment)
#     # use x1,y1 and x2,y2 as the two points on the segment
#     x1, y1 = segment[1]
#     x2, y2 = segment[2]
#     # use x0, y0 as the target point
#     x0, y0 = target_pt

#     # make a line out of the two points on the segment
#     # the line is ax + by + c = 0
#     a = y2 - y1
#     b = x1 - x2
#     c = x2 * y1 - x1 * y2
#     # on this line, find a point that is closest to the target point
#     # the point is x = (b(bx0 - ay0) - ac) / (a^2 + b^2)
#     # the point is y = (a(-bx0 + ay0) - bc) / (a^2 + b^2)
#     if a ** 2 + b ** 2 == 0:
#         closest_x = x1
#         closest_y = y1
#     else:
#         closest_x = (b * (b * x0 - a * y0) - a * c) / (a ** 2 + b ** 2)
#         closest_y = (a * (-b * x0 + a * y0) - b * c) / (a ** 2 + b ** 2)


#     # check if the point is within the segment
#     # the point is within the segment if it is between the two points
#     if min(x1, x2) <= closest_x <= max(x1, x2) and min(y1, y2) <= closest_y <= max(y1, y2):
#         # if it is, return the distance between the point and the target point
#         return ((closest_x - x0) ** 2 + (closest_y - y0) ** 2) ** 0.5
#     else:
#         # if it is not, return the distance between the target point and the closest end-point on the segment
#         return min(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5, ((x2 - x0) ** 2 + (y2 - y0) ** 2) ** 0.5)


def line_dist_to_pt(segment, target_pt):
    """
    Calculate the distance from a line segment to a target point.
    Handles floating-point precision issues using epsilon.
    """
    epsilon = 1e-10  # Small tolerance for floating-point comparisons

    # Points on the segment
    x1, y1 = segment[1]
    x2, y2 = segment[2]

    # Target point
    x0, y0 = target_pt

    # Line coefficients: ax + by + c = 0
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2

    # Denominator for closest point calculation
    denominator = a**2 + b**2

    # Handle degenerate segment (zero length)
    if denominator < epsilon:
        # If the segment is essentially a point, return distance to that point
        return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5

    # Closest point on the infinite line
    closest_x = (b * (b * x0 - a * y0) - a * c) / denominator
    closest_y = (a * (-b * x0 + a * y0) - b * c) / denominator

    # Check if the closest point lies on the segment, with epsilon tolerance
    if (
        min(x1, x2) - epsilon <= closest_x <= max(x1, x2) + epsilon
        and min(y1, y2) - epsilon <= closest_y <= max(y1, y2) + epsilon
    ):
        # Return the distance from the target point to the closest point on the segment
        return ((closest_x - x0) ** 2 + (closest_y - y0) ** 2) ** 0.5
    else:
        # Return the distance to the closest endpoint of the segment
        dist_to_start = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        dist_to_end = ((x2 - x0) ** 2 + (y2 - y0) ** 2) ** 0.5
        return min(dist_to_start, dist_to_end)


# ================== logics about circles ==================


# distance canonicalized circle to a point
def circle_dist_to_pt(circle, target_pt):
    circle_type, origin, radius = circle
    # print (circle)
    # get the distance between the center and the target point
    dist_center = (
        (origin[0] - target_pt[0]) ** 2 + (origin[1] - target_pt[1]) ** 2
    ) ** 0.5
    # return the difference between the distance and the radius
    return abs(dist_center - radius)


# ================== logic about arcs ==================


# distance from canonicalized arc to a point
def arc_dist_to_pt(arc, target_pt):
    arc_type, arc_center, arc_start, arc_mid, arc_end = arc

    # get the vector offset from center to start, center to mid, center to end
    v_c_s = (arc_start[0] - arc_center[0], arc_start[1] - arc_center[1])
    v_c_m = (arc_mid[0] - arc_center[0], arc_mid[1] - arc_center[1])
    v_c_e = (arc_end[0] - arc_center[0], arc_end[1] - arc_center[1])

    # get the vector from center to target
    v_c_t = (target_pt[0] - arc_center[0], target_pt[1] - arc_center[1])

    # get the angle of the vectors
    angle_c_s = np.arctan2(v_c_s[1], v_c_s[0])
    angle_c_m = np.arctan2(v_c_m[1], v_c_m[0])
    angle_c_e = np.arctan2(v_c_e[1], v_c_e[0])
    angle_c_t = np.arctan2(v_c_t[1], v_c_t[0])

    expanded_list = [
        ("s", angle_c_s),
        ("m", angle_c_m),
        ("e", angle_c_e),
        ("t", angle_c_t),
    ]
    # add 2pi to everything in the expanded list
    expanded_list_2pi = [(name, angle + 2 * np.pi) for name, angle in expanded_list]
    together = expanded_list + expanded_list_2pi

    # sort the list by angle
    together.sort(key=lambda x: x[1])
    # convert it into a string of letters
    order_str = "".join([x[0] for x in together])
    # replace 'm' and 't' with 'x'
    order_str = order_str.replace("m", "x").replace("t", "x")

    pt_on_arc = "sxxe" in order_str or "exxs" in order_str

    # if the point is on the arc, return the distance from the point to the circle of the arc
    if pt_on_arc:
        radius = (v_c_s[0] ** 2 + v_c_s[1] ** 2) ** 0.5
        dist_target_center = (v_c_t[0] ** 2 + v_c_t[1] ** 2) ** 0.5
        return abs(dist_target_center - radius)
    else:
        # otherwise return the distance from the point to the closest end-point of the arc
        dist_start = (
            (arc_start[0] - target_pt[0]) ** 2 + (arc_start[1] - target_pt[1]) ** 2
        ) ** 0.5
        dist_end = (
            (arc_end[0] - target_pt[0]) ** 2 + (arc_end[1] - target_pt[1]) ** 2
        ) ** 0.5
        return min(dist_start, dist_end)


# ================== give sdf on a grid ==================
def get_sdf(scene_canonicalized_curves, canvas_w, n_steps):
    step_size = canvas_w / n_steps
    grid = np.zeros((n_steps, n_steps))
    for i in range(n_steps):
        for j in range(n_steps):
            target_x, target_y = i * step_size, j * step_size
            min_dist = float("inf")
            for curve in scene_canonicalized_curves:
                curve_type, *curve_data = curve
                if curve_type == "line":
                    dist = line_dist_to_pt(curve, (target_x, target_y))
                elif curve_type == "circle":
                    dist = circle_dist_to_pt(curve, (target_x, target_y))
                elif curve_type == "arc":
                    dist = arc_dist_to_pt(curve, (target_x, target_y))
                min_dist = min(min_dist, dist)
            grid[j, i] = min_dist
    return grid


# ================== visualization ==================


# small helper to convert scene into a render format understood by visualize.py
def convert_scene_for_vis(scene_parsed_curves):
    ret = []
    for curve in scene_parsed_curves:
        curve_type, *curve_data = curve
        if curve_type == "line":
            # add to ret a form of line [pt1, pt2]
            ret.append([curve_data[0], curve_data[1]])
        if curve_type == "circle":
            # get the two points from the circle
            pt1 = curve_data[0]
            pt2 = curve_data[1]
            # find the center of the circle
            center = ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)
            radius = ((pt1[0] - center[0]) ** 2 + (pt1[1] - center[1]) ** 2) ** 0.5
            # create 4 points equal distance on the circle
            ptA = (center[0] + radius, center[1])
            ptB = (center[0], center[1] + radius)
            ptC = (center[0] - radius, center[1])
            ptD = (center[0], center[1] - radius)
            ret.append([ptA, ptB, ptC, ptD])
        if curve_type == "arc":
            # add to ret a form of arc [start, mid, end]
            ret.append([curve_data[0], curve_data[1], curve_data[2]])
    return {"entities": ret}


# ================== sample points from the curves ==================


# sample n_curve_pts points on the line segment
def sample_from_line(segment):
    # use x1,y1 and x2,y2 as the two points on the segment
    x1, y1 = segment[1]
    x2, y2 = segment[2]
    # sample n_curve_pts points on the line, including the two end points
    # use raw python, no numpy
    curve_pts = []
    for i in range(N_CURVE_PTS):
        x = x1 + (((x2 - x1) * i) / (N_CURVE_PTS - 1))
        y = y1 + (((y2 - y1) * i) / (N_CURVE_PTS - 1))
        curve_pts.append((x, y))
    # assert both end points are the same as the segment's end points

    # print(curve_pts[0], (x1, y1))
    # print(curve_pts[-1], (x2, y2))

    # assert curve_pts[0] == (x1, y1)
    # assert curve_pts[-1] == (x2, y2)
    return curve_pts


# sample n_curve_pts points on the circle
def sample_from_circle(circle):
    circle_type, origin, radius = circle
    # sample n_curve_pts points on the circle
    # use raw python, no numpy
    curve_pts = []
    for i in range(N_CURVE_PTS):
        angle = (2 * np.pi * i) / N_CURVE_PTS
        x = origin[0] + (radius * np.cos(angle))
        y = origin[1] + (radius * np.sin(angle))
        curve_pts.append((x, y))
    return curve_pts


# sample n_curve_pts points on the arc
def sample_from_arc(arc):
    arc_type, arc_center, arc_start, arc_mid, arc_end = arc
    # get the start angle
    start_angle = np.arctan2(arc_start[1] - arc_center[1], arc_start[0] - arc_center[0])
    # get the end angle
    end_angle = np.arctan2(arc_end[1] - arc_center[1], arc_end[0] - arc_center[0])
    # get the mid angle
    mid_angle = np.arctan2(arc_mid[1] - arc_center[1], arc_mid[0] - arc_center[0])

    # check if start > end, if so, add 2pi to end
    if start_angle > end_angle:
        end_angle += 2 * np.pi
    # sample n_curve_pts points on the arc
    # use raw python, no numpy
    curve_pts = []
    for i in range(N_CURVE_PTS):
        angle = start_angle + (end_angle - start_angle) * i / (N_CURVE_PTS - 1)
        x = arc_center[0] + (
            np.cos(angle)
            * np.sqrt(
                (arc_start[0] - arc_center[0]) ** 2
                + (arc_start[1] - arc_center[1]) ** 2
            )
        )
        y = arc_center[1] + (
            np.sin(angle)
            * np.sqrt(
                (arc_start[0] - arc_center[0]) ** 2
                + (arc_start[1] - arc_center[1]) ** 2
            )
        )
        curve_pts.append((x, y))
    return curve_pts


def dist_chamfer_summed_asymmetric(scene_A, scene_B, max_point_dist=5):

    scene_canonicalized_A = canonicalize_curves(scene_A)
    scene_canonicalized_B = canonicalize_curves(scene_B)

    # print('scA', scene_canonicalized_A)
    # print('scB', scene_canonicalized_B)

    # for each curve in A, sample points from it
    A_points = []
    for curve in scene_canonicalized_A:
        if curve[0] == "line":
            A_points.extend(sample_from_line(curve))
        elif curve[0] == "circle":
            A_points.extend(sample_from_circle(curve))
        elif curve[0] == "arc":
            A_points.extend(sample_from_arc(curve))
            # pritn('points from arc', curve, sample_from_arc(curve))

    min_dists = []
    # use distance_sdf logic to get the sdf of A_points to B
    for pt in A_points:
        target_x, target_y = pt
        min_dist = max_point_dist
        for curve in scene_canonicalized_B:
            if curve[0] == "line":
                dist = line_dist_to_pt(curve, pt)
            elif curve[0] == "circle":
                dist = circle_dist_to_pt(curve, pt)
            elif curve[0] == "arc":
                dist = arc_dist_to_pt(curve, pt)
            if dist < min_dist:
                min_dist = dist
        min_dists.append(min_dist)

    summed_distance = np.sum(min_dists)

    maximum_summed_distance = max_point_dist * len(A_points)

    if len(A_points) == 0:
        return 1
    else:
        proportional_distance = summed_distance / maximum_summed_distance

        # return 1000 if nan
        if math.isnan(proportional_distance):
            print("nan distance found for", scene_A, scene_B)
            return 1
        else:
            return proportional_distance


def dist_chamfer_summed_symmetric(
    scene_A, scene_B, canvas_size=41, max_distance_proportion=0.25
):

    # if all elements exist but are over a certain distance away, you're still maximally far away
    # if you add half the elements and they're zero distance away, you're halfway

    # you can't divide by the total number of elements in the target (or total n of points in a target), because you can reconstruct the target with fewer elements

    # the sum is a sum of how far away you are, with elements directly on top of the target geoms being 0 distance away
    # but what if you're missing half of the design?
    # all points are are 0 away in one direction
    # in the other direction, 1/2 your points will have zero distance. The other half will have a range of distances (very few will be maximum).

    max_point_dist = (canvas_size - 1) * (max_distance_proportion)

    return (
        dist_chamfer_summed_asymmetric(scene_A, scene_B, max_point_dist=max_point_dist)
        + dist_chamfer_summed_asymmetric(
            scene_B, scene_A, max_point_dist=max_point_dist
        )
    ) / 2


def center_geometries(curves, y_offset=20, x_offset=20):
    """
    Centers the geometries in the scene by adjusting the x and y coordinates of the points.
    """
    centered_curves = []
    for curve in curves:
        curve_type, *points = curve
        centered_points = []
        for point in points:
            i, j = point
            centered_points.append((i + y_offset, j + x_offset))
        centered_curves.append((curve_type, *centered_points))
    return centered_curves


def parse_stim_design(target):
    curves = target["design"]["curves"]
    parsed = []
    for c in curves:
        parsed.append(tuple([c["type"]] + [(a, b) for (a, b) in c["control_points"]]))
    return parsed


def flip_geometries(curves):
    """
    Flip geometries in y axis
    """
    flipped_curves = []
    for curve in curves:
        curve_type, *points = curve
        flipped_points = []
        for point in points:
            i, j = point
            flipped_points.append((i, -j))
        flipped_curves.append((curve_type, *flipped_points))
    return flipped_curves
