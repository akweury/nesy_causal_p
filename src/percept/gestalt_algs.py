# Created by jing at 12.01.25
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


def algo_closure_position(args, input_groups):
    """ group input groups to output groups, which are high level groups """
    # each object assigned a group id as its label
    args.obj_fm_size = 32

    points = np.stack([group.pos for group in input_groups])
    assigned_mask = np.zeros(len(points), dtype=bool)
    all_lines = []
    all_arcs = []
    max_iterations = 5
    for iteration in range(max_iterations):
        # Step 1: compute hull of unassigned points
        unassigned_indices = [i for i in range(len(points)) if not assigned_mask[i]]
        unassigned_points = points[unassigned_indices]
        if len(unassigned_points) == 0:
            break
        group_lines = percept_lines.detect_lines(tuple(unassigned_points.tolist()))
        group_lines = percept_lines.update_lines(group_lines, tuple(unassigned_points.tolist()))
        group_arcs = []

        all_lines.extend(group_lines)
        all_arcs.extend(group_arcs)
        assigned_mask = update_assigned_mask(len(points), all_lines)
    #
    hull_imgs = []
    for gl_i in range(len(all_lines)):
        hull_imgs.append(chart_utils.show_convex_hull(np.array(points), np.array(all_lines[gl_i]["points"])))
    if len(hull_imgs) > 0:
        chart_utils.van(hull_imgs)

    lines_data = percept_lines.get_line_data(all_lines)
    # find polygons or circles
    labels = torch.zeros(len(input_groups))
    group_labels = []
    line_group_data = [line["indices"] for line in all_lines]
    labels, hasTriangle, group_labels = models.find_triangles(lines_data, line_group_data, labels, group_labels)
    labels, hasSquare, group_labels = models.find_squares(lines_data, line_group_data, labels, group_labels)
    return labels, group_labels


def algo_closure(args, segments):
    """ group input groups to output groups, which are high level groups """
    # each object assigned a group id as its label
    bk_shapes = bk.load_bk_fms(args, bk.bk_shapes)
    args.obj_fm_size = 32
    img = data_utils.merge_segments(segments)

    labels = torch.zeros(len(segments))
    label_counter = 1
    group_label = 0
    contour_points, contour_segs, contour_seg_labels = models.get_contour_segs(img)

    # base on the segments, what shape can you recall by considering closure principle
    # find line groups
    lines, line_group_data = models.get_line_groups(contour_points, contour_segs, contour_seg_labels, img.shape[0])
    curves = models.get_curves(contour_points, contour_segs, contour_seg_labels, img.shape[0])

    normed_similar_lines = []
    for line in lines:
        normed_similar_lines.append([line[0], line[1].astype(np.float32) / 1024, line[2].astype(np.float32) / 1024])

    obj_labels = torch.zeros(len(segments))
    group_labels = []
    # find polygons or circles
    labels, hasTriangle, group_labels = models.find_triangles(normed_similar_lines, line_group_data, obj_labels,
                                                              group_labels)
    labels, hasSquare, group_labels = models.find_squares(normed_similar_lines, line_group_data, obj_labels,
                                                          group_labels)
    return labels, group_labels


def cluster_by_closure(args, segments, obj_groups):
    all_labels = []
    group_lengths = []

    all_shapes = []
    for example_i in range(len(segments)):
        segment = segments[example_i]
        obj_group = obj_groups[example_i]
        if len(obj_group) > 5:
            labels, shapes = algo_closure_position(args, obj_group)
        else:
            labels, shapes = algo_closure(args, segment)

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
