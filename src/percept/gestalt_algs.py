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
        shapes.append(len(np.unique(preds)) * [0])
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
    labels, hasTriangle, group_labels = models.find_triangles(lines_data, all_lines, labels, group_labels)
    labels, hasSquare, group_labels = models.find_squares(lines_data, all_lines, labels, group_labels)
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
    lines = models.get_line_groups(contour_points, contour_segs, contour_seg_labels, img.shape[0])
    curves = models.get_curves(contour_points, contour_segs, contour_seg_labels, img.shape[0])
    # find polygons or circles
    triangles = models.find_triangles(lines)
    squares = models.find_squares(lines)
    # circles = models.find_circles(curves)
    if triangles:
        group_label = 1
    elif squares:
        group_label = 2
    else:
        group_label = 3
    labels = labels + 1

    return labels, group_label


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
