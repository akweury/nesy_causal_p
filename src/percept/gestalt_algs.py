# Created by jing at 12.01.25
import numpy as np
import torch
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from src.utils import data_utils
from src import bk
from src.utils.chart_utils import van
from src.memory import recall
from src.neural import models
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
    for ocm in ocms:
        pred = algo_proximity(ocm[:, :2], th)
        preds.append(pred)

    return preds, th


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
    for ocm in ocms:
        pred = algo_similarity(ocm, mode)
        th_clusters.append(len(pred.unique()))
        preds.append(pred)
    return preds


def algo_closure(args, segments, input_groups):
    """ group input groups to output groups, which are high level groups """
    # each object assigned a group id as its label
    bk_shapes = bk.load_bk_fms(args, bk.bk_shapes)
    args.obj_fm_size = 32
    # all_obj_found_labels = False
    img = data_utils.merge_segments(segments)

    # preprocessing img, convert rgb image to black-white image
    # cropped_img, crop_data = data_utils.crop_img(img)
    # bw_img = data_utils.resize_img(cropped_img, resize=args.obj_fm_size).unsqueeze(0)
    # groups = []
    labels = torch.zeros(len(segments))
    label_counter = 1
    group_label = 0
    while torch.any(labels[:len(input_groups)] == 0):
        # recall the memory
        memory, group_label, cropped_data, kernels = recall.recall_match(args, bk_shapes, img)
        # assign each object a label
        same_group = reason.reason_labels(input_groups, cropped_data, labels, memory, kernels)

        if same_group.sum() == 0:
            break
        labels[same_group] += label_counter
        label_counter += 1
    return labels, group_label


def cluster_by_closure(args, segments, obj_groups):
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

    same_group_num = len(torch.tensor(group_lengths).unique()) == 1
    same_group_label = len(torch.tensor(all_shapes).unique()) == 1
    if same_group_num and same_group_label:
        pred_label = all_labels[0]
        pred_shape = all_shapes[0]
    else:
        pred_label = None
        pred_shape = None
    return pred_label, pred_shape
