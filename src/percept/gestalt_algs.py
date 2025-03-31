# Created by x at 12.01.25
import numpy as np
import torch
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from src.utils import data_utils
from src import bk
from src.utils import chart_utils
from src.memory import recall
from src.neural import models, merge_lines, percept_lines
from src.reasoning import reason


def proximity_distance(u, v):
    return np.linalg.norm(u - v)


def similarity_distance(a, b, mode):
    """Compute weighted distance between object i and j."""
    a_dict = bk.tensor2dict(a)
    b_dict = bk.tensor2dict(b)
    shape_i = a_dict["shape"]
    shape_j = b_dict["shape"]
    color_i = a_dict["color"]
    color_j = b_dict["color"]

    # Shape distance
    if mode == "shape":
        dist = 0 if torch.all(shape_i == shape_j) else 1
    else:
        dist = 0 if torch.all(color_i == color_j) else 1

    return dist


def average_pairwise_distance(points):
    # Convert list of points to a PyTorch tensor
    points_tensor = torch.tensor(points, dtype=torch.float32)

    # Compute pairwise differences
    diff = points_tensor.unsqueeze(0) - points_tensor.unsqueeze(1)

    # Compute Euclidean distances
    distances = torch.norm(diff, dim=2)

    # Extract upper triangle (excluding diagonal) to avoid redundant pairs
    num_points = points_tensor.shape[0]
    triu_indices = torch.triu_indices(num_points, num_points, offset=1)
    pairwise_distances = distances[triu_indices[0], triu_indices[1]]

    # Calculate and return the average distance
    return pairwise_distances.mean().item()


def algo_proximity(ocm, th):
    obj_n = ocm.shape[0]
    labels = torch.full((obj_n,), 0, dtype=torch.int32)
    visited = torch.zeros(obj_n, dtype=torch.bool)
    current_label = 0
    for i in range(obj_n):
        if not visited[i]:
            # BFS or DFS
            stack = [i]
            visited[i] = True
            labels[i] = current_label
            while stack:
                top = stack.pop()
                for j in range(obj_n):
                    if not visited[j]:
                        dist = proximity_distance(ocm[top, :2], ocm[j, :2])
                        if dist <= th:
                            visited[j] = True
                            labels[j] = current_label
                            stack.append(j)
            current_label += 1
    return labels


def algo_symmetry(points, tol=0.05):
    """
    Check if an Nx2 numpy array of points (with x, y in [0,1]) is roughly symmetric
    about the vertical axis x = 0.5, and return labels for each point.

    A point is considered symmetric if:
      - It lies close to the vertical axis (|x - 0.5| <= tol), or
      - There is another point whose coordinates are close to (1 - x, y) (within tol).

    Additionally, each point is labeled as:
      - 0 if x < 0.5 (left side)
      - 1 if x >= 0.5 (right side)

    Args:
        points (np.ndarray): An Nx2 array of points.
        tol (float): Tolerance for symmetry matching.

    Returns:
        tuple:
          - is_symmetric (bool): True if the points are roughly symmetric.
          - labels (np.ndarray): An array of length N with values 0 (left) or 1 (right).
    """
    # Compute labels using vectorized operations:
    labels = np.where(points[:, 0] < 0.5, 0, 1)

    # For the symmetry check, work with a list (so we can remove matched points)
    # Each point is represented as a [x, y] list.
    points_list = points.tolist()

    while points_list:
        # Pop the first point from the list
        x, y = points_list.pop(0)

        # If the point lies near the vertical axis, it's symmetric on its own.
        if abs(x - 0.5) <= tol:
            continue

        # Calculate the expected mirror point: (1 - x, y)
        mirror_x = 1 - x

        # Search for a matching mirror point in the remaining points.
        found_match = False
        for i, (xx, yy) in enumerate(points_list):
            if abs(xx - mirror_x) <= tol and abs(yy - y) <= tol:
                found_match = True
                # Remove the mirror point so that it isn't reused.
                points_list.pop(i)
                break

        # If no mirror is found, the points are not symmetric.
        if not found_match:
            return False, labels

    # All points have either been matched or lie close to the symmetry axis.
    return True, labels


def cluster_by_proximity(ocms):
    """ Function to compute distance or difference
    Return:
        labels 1 x O np array
        groups 1 x O x P np array
    """

    th = 0.2
    preds = []
    shapes = []
    for ocm in ocms:
        pred = algo_proximity(ocm[:, :2], th)
        preds.append(pred)
        shapes.append(len(np.unique(pred)) * [0])
    return preds, th, shapes


def algo_similarity(ocm, mode):
    obj_n = ocm.shape[0]
    labels = torch.full((obj_n,), 0, dtype=torch.int32)
    visited = torch.zeros(obj_n, dtype=torch.bool)
    current_label = 1
    for i in range(obj_n):
        if not visited[i]:
            # BFS or DFS
            stack = [i]
            visited[i] = True
            labels[i] = current_label
            while stack:
                top = stack.pop()
                for j in range(obj_n):
                    if not visited[j]:
                        dist = similarity_distance(ocm[top], ocm[j], mode)
                        if dist == 0:
                            visited[j] = True
                            labels[j] = current_label
                            stack.append(j)
            current_label += 1
    return labels


def cluster_by_similarity(ocms, mode):
    th_clusters = []
    preds = []
    shapes = []
    for ocm in ocms:
        pred = algo_similarity(ocm, mode)
        th_clusters.append(len(pred.unique()))
        preds.append(pred)
        shapes.append(len(np.unique(pred)) * [0])
    return preds, shapes


def update_assigned_mask(point_num, all_lines):
    assigned_mask = np.zeros(point_num, dtype=bool)
    assigned_mask[:] = False
    for line in all_lines:
        assigned_mask[list(line["indices"])] = True
    return assigned_mask


import math
import itertools

import math
import itertools


def round_list(l, n):
    return [round(value, n) for value in l]


def find_lines_from_points(points, tolerance=0.05, min_points=3, slope_th=0.1):
    """
    Given a list of 2D points, detects lines by grouping nearly-collinear points
    using an error tolerance. A point may belong to more than one line if lines
    intersect. Each detected line is returned as a list containing:
      [slope, endpoint1, endpoint2],
    where endpoint1 and endpoint2 are the extreme points among the inliers along the line.

    The algorithm works as follows:
      1. For every unique pair of points, compute the line in normalized form:
           a*x + b*y + c = 0,
         where (a, b) is the unit normal vector. We enforce a >= 0 for a unique representation.
      2. For the candidate line, count all points whose perpendicular distance
         from the line is within the given tolerance. These are the inliers.
      3. If the number of inliers is at least min_points (the minimum number of points required to form a line),
         project the inlier points onto the line’s direction (which is perpendicular to the normal) and compute
         the two extreme projections. These become the endpoints of the line segment.
      4. Compute the line's slope for output. For non-vertical lines (when |b| > 1e-6),
         the slope is -a/b; otherwise it is considered vertical (represented as float('inf')).
      5. To avoid duplicates (different pairs yielding essentially the same line),
         a new candidate is only added if its normalized (a, b, c) parameters differ
         from those of previously detected lines by at least the tolerance.

    Parameters:
      points (list): List of 2D points, where each point is a tuple (x, y).
      tolerance (float): Maximum distance from the candidate line for a point to be
                         considered an inlier, and also used for comparing line parameters.
      min_points (int): Minimum number of inlier points required to accept a line.

    Returns:
      list: A list of detected lines. Each line is represented as:
            [slope, endpoint1, endpoint2]
    """
    detected_lines = []
    detected_lines_indices = []
    n = len(points)

    # Iterate over every unique pair of points
    for i, j in itertools.combinations(range(n), 2):
        p1 = points[i]
        p2 = points[j]
        # Skip nearly identical points
        if math.isclose(p1[0], p2[0], abs_tol=tolerance) and math.isclose(p1[1], p2[1], abs_tol=tolerance):
            continue

        # Compute the line parameters in the form a*x + b*y + c = 0.
        # Using two-point form: a = y2 - y1, b = x1 - x2.
        a = p2[1] - p1[1]
        b = p1[0] - p2[0]
        norm = math.hypot(a, b)
        if norm == 0:
            continue
        a /= norm
        b /= norm
        c = -(a * p1[0] + b * p1[1])
        # Enforce a unique representation by ensuring a is non-negative.
        if a < 0:
            a, b, c = -a, -b, -c

        # Determine inliers: points whose distance to the line is within tolerance.
        inliers = []
        inliers_indices = []
        for p_i, p in enumerate(points):
            distance = abs(a * p[0] + b * p[1] + c)
            if distance <= tolerance:
                inliers.append(p)
                inliers_indices.append(p_i)

        # Use the argument min_points to determine if enough inliers are present.
        if len(inliers) < min_points:
            continue

        # Compute endpoints of the line segment from the inlier set.
        # The line's direction vector is perpendicular to the normal: d = (-b, a).
        dvec = (-b, a)
        # Project each inlier onto the direction vector.
        projections = [p[0] * dvec[0] + p[1] * dvec[1] for p in inliers]
        min_proj = min(projections)
        max_proj = max(projections)
        endpoint1 = round_list(inliers[projections.index(min_proj)], 2)
        endpoint2 = round_list(inliers[projections.index(max_proj)], 2)

        # Compute slope for output.
        # For non-vertical lines, slope is computed as -a/b.
        if abs(b) > 5 * 1e-2:
            slope = round(-a / b / slope_th) * slope_th
        else:
            slope = 10
        # Check if a similar line has already been detected.
        duplicate_found = False
        for (a2, b2, c2, _, _, _) in detected_lines:
            if abs(a - a2) < tolerance and abs(b - b2) < tolerance and abs(c - c2) < tolerance:
                duplicate_found = True
                break
        if duplicate_found:
            continue

        if endpoint1[0] < endpoint2[0]:
            detected_lines.append((a, b, c, slope, endpoint1, endpoint2))
        else:
            detected_lines.append((a, b, c, slope, endpoint2, endpoint1))
        detected_lines_indices.append(inliers_indices)
    # Return the detected lines in the specified format.
    lines = [[slope, endpoint1, endpoint2] for (_, _, _, slope, endpoint1, endpoint2) in detected_lines]
    unique_lines = []
    unique_lines_indices = []
    for l_i, line in enumerate(lines):
        if line not in unique_lines:
            unique_lines.append(line)
            unique_lines_indices.append(detected_lines_indices[l_i])
    return unique_lines, unique_lines_indices


def algo_find_arcs(points):
    all_arcs = []
    return all_arcs


def algo_closure_position(args, input_groups):
    """ group input groups to output groups, which are high level groups """
    # each object assigned a group id as its label
    args.obj_fm_size = 32
    # find polygons or circles
    labels = torch.zeros(len(input_groups))
    group_labels = []

    points = np.stack([group.pos.to("cpu") for group in input_groups])
    assigned_mask = np.zeros(len(points), dtype=bool)
    all_lines, line_points = find_lines_from_points(points.tolist(), tolerance=0.001, min_points=args.line_min_size)
    all_arcs = algo_find_arcs(points)

    # hull_imgs = []
    # for gl_i in range(len(all_lines)):
    #     hull_imgs.append(chart_utils.show_convex_hull(np.array(points), np.array(all_lines[gl_i]["points"])))
    # if len(hull_imgs) > 0:
    #     chart_utils.van(hull_imgs)

    # lines_data = percept_lines.get_line_data(all_lines)

    # line_group_data = [line["indices"] for line in all_lines]

    # determine which lines form the shape of triangles
    labels, group_labels = models.find_position_closure_triangles(all_lines, line_points, labels, group_labels)
    labels, group_labels = models.find_position_closure_squares(all_lines, line_points, labels, group_labels)

    return labels, group_labels


def find_center_of_segment(segment):
    segment[segment == [211, 211, 211]] = 0
    nonzero_indices = torch.argwhere(torch.any(segment > 0, dim=-1))
    center_y, center_x = torch.mean(nonzero_indices.float(), dim=0)

    return (int(center_x), int(center_y))


def algo_closure(args, segments, obj_groups):
    """ group input groups to output groups, which are high level groups """
    # each object assigned a group id as its label
    bk_shapes = bk.load_bk_fms(args, bk.bk_shapes)
    args.obj_fm_size = 32
    img = data_utils.merge_segments(segments)

    labels = torch.zeros(len(segments))
    label_counter = 1
    group_label = 0
    contour_points, contour_segs, contour_seg_labels = models.get_contour_segs(img)

    obj_centers = [group.pos for group in obj_groups]
    # base on the segments, what shape can you recall by considering closure principle
    # find line groups
    rectangle_lines = models.get_line_groups(contour_points, contour_segs, contour_seg_labels, img.shape[0])
    triangle_lines = models.get_triangle_groups(contour_points, contour_segs, contour_seg_labels, img.shape[0])
    curves = models.get_curves(contour_points, contour_segs, contour_seg_labels, img.shape[0])

    # normed_similar_lines = []
    # for line in lines:
    #     normed_similar_lines.append([line[0], line[1].astype(np.float32) / 1024, line[2].astype(np.float32) / 1024])

    obj_labels = torch.zeros(len(segments))

    group_labels = []
    # find polygons or circles
    labels, group_labels = models.find_triangles(triangle_lines, obj_centers, obj_labels,
                                                 group_labels)
    labels, group_labels = models.find_squares(rectangle_lines, obj_centers, obj_labels,
                                               group_labels)
    return labels, group_labels


def cluster_by_feature_closure(args, segments, obj_groups):
    all_labels = []
    group_lengths = []

    all_shapes = []
    for example_i in range(len(segments)):
        segment = segments[example_i]
        obj_group = obj_groups[example_i]
        labels, shapes = algo_closure(args, segment, obj_group)
        group_lengths.append(len(labels.unique()))
        all_labels.append(labels)
        all_shapes.append(shapes)

    return all_labels, all_shapes
def cluster_by_position_closure(args, obj_groups):
    all_labels = []
    group_lengths = []

    all_shapes = []
    for example_i in range(len(obj_groups)):
        obj_group = obj_groups[example_i]
        labels, shapes = algo_closure_position(args, obj_group)
        group_lengths.append(len(labels.unique()))
        all_labels.append(labels)
        all_shapes.append(shapes)

    return all_labels, all_shapes



def cluster_by_symmetry(ocms):
    th = 0.05
    preds = []
    shapes = []
    for ocm in ocms:
        is_symmetry, pred = algo_symmetry(ocm[:, :2].numpy(), th)
        preds.append(pred)
        shapes.append(len(np.unique(pred)) * [0])
    return preds, th, shapes
