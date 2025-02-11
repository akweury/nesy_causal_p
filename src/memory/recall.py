# Created by x at 10.12.24
import numpy as np
import torch

from src.memory.recall_utils import *
from src.utils import data_utils, chart_utils
from src.neural import models
from src.reasoning import reason
from src.utils.chart_utils import van
from PIL import Image, ImageDraw
import itertools
import cv2
from skimage.transform import resize
import numpy as np
from scipy.ndimage import label, find_objects, center_of_mass
from skimage.draw import line
from itertools import combinations
from src import bk
import matplotlib.pyplot as plt
import math


def recall_fms(input_fms, bk_fms, reshape=None):
    # scores = []
    # recall_fm = []
    # convolutional layer
    # input_fms = models.img2fm(img, bk_shapes["kernels"].float())
    # bk_fms = bk_shapes["fm_repo"]
    # fm similarity

    # Flatten the tensors along the spatial dimensions
    first_tensor_flat = input_fms.view(1, -1)  # Shape: 1x(Channel*16*16)
    second_tensor_flat = bk_fms.view(bk_fms.shape[0], -1)  # Shape: Nx(Channel*16*16)

    # Compute cosine similarity between first_tensor and all tensors in second_tensor
    cosine_similarities = F.cosine_similarity(first_tensor_flat, second_tensor_flat)
    most_similar_score = cosine_similarities.max().item()

    # cosine_similarities.mean()
    # Find the index of the most similar tensor
    most_similar_index = torch.argmax(cosine_similarities).item()
    # chart_utils.show_line_chart(sorted(cosine_similarities))

    # Get the most similar tensor

    most_similar_fm = bk_fms[most_similar_index]

    # sim_total = data_utils.matrix_equality(bk_fms, input_fms.unsqueeze(0))
    # scores = sim_total.max()
    # recall_fm = bk_fms[sim_total.argmax()]
    # best_shape = np.argmax(scores)
    # best_recall_fm = recall_fm[best_shape]
    return most_similar_fm, most_similar_score


def crop_to_valid_area(image, threshold=10):
    """
    Crop the image to the smallest bounding box that contains valid information.

    Args:
        image (np.ndarray): Grayscale test image as a NumPy array.
        threshold (int): Pixel intensity threshold to identify non-empty regions.

    Returns:
        np.ndarray: Cropped image.
        tuple: (x_min, y_min, x_max, y_max) coordinates of the cropped region.
    """
    # Find non-zero areas
    valid_rows = np.any(image > threshold, axis=1)
    valid_cols = np.any(image > threshold, axis=0)

    if not np.any(valid_rows) or not np.any(valid_cols):
        # No valid area
        return image, (0, 0, image.shape[1], image.shape[0])

    y_min, y_max = np.where(valid_rows)[0][[0, -1]]
    x_min, x_max = np.where(valid_cols)[0][[0, -1]]

    # Include bordering cases by padding
    padding = 0  # Add padding to account for partial patches
    y_min = max(0, y_min - padding)
    y_max = min(image.shape[0], y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(image.shape[1], x_max + padding)

    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image, (x_min, y_min, x_max, y_max)


def connect_isolated_areas(image):
    """
    Connect isolated areas in a black-and-white image.

    Args:
        image (np.ndarray): Black-and-white image (2D array) with valid values as non-zero.

    Returns:
        np.ndarray: Modified image with connections between isolated areas.
    """
    # Step 1: Identify isolated areas using connected component labeling
    image[image > 0] = 1
    labeled_image, num_features = label(image > 0)  # Non-zero values are valid
    if num_features <= 1:
        return image  # Nothing to connect

    # Step 2: Compute centroids of all isolated areas
    centroids = center_of_mass(image, labeled_image, range(1, num_features + 1))

    # Step 3: Find nearest neighbors for each centroid and connect them
    connected_image = image.copy()
    for i, centroid1 in enumerate(centroids):
        min_distance = float("inf")
        nearest_centroid = None

        # Find the nearest centroid to the current centroid
        for j, centroid2 in enumerate(centroids):
            if i == j:
                continue
            distance = np.linalg.norm(np.array(centroid1) - np.array(centroid2))
            if distance < min_distance:
                min_distance = distance
                nearest_centroid = centroid2

        # Step 4: Draw a line between the two centroids
        if nearest_centroid:
            rr, cc = line(int(centroid1[0]), int(centroid1[1]), int(nearest_centroid[0]), int(nearest_centroid[1]))
            connected_image[rr, cc] = 255  # Set line pixels to a valid value

    return connected_image


from scipy.spatial.distance import cdist


def find_contour(img):
    binary_image = (img > 0).astype(np.uint8) * 255
    smoothed_image = cv2.GaussianBlur(binary_image, (5, 5), 0)  # Gaussian smoothing

    # Step 2: Detect edges using Canny edge detection
    edges = cv2.Canny(smoothed_image, threshold1=50, threshold2=150)

    # Step 3: Extract contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 4: Approximate the polygon
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)  # Smaller epsilon for detailed approximation
    polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Step 5: Sub-pixel refinement of the corners
    refined_corners = cv2.cornerSubPix(smoothed_image, np.float32(polygon).reshape(-1, 1, 2), (5, 5), (-1, -1),
                                       criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))

    # Step 6: Detect edges between refined points
    refined_corners = refined_corners.squeeze()
    edges = []
    for i in range(len(refined_corners)):
        x1, y1 = refined_corners[i]
        x2, y2 = refined_corners[(i + 1) % len(refined_corners)]  # Connect last to first
        edges.append((x1, y1, x2, y2))

    # Step 7: Draw the refined polygon for visualization
    contour_image = np.zeros_like(img)
    for edge in edges:
        x1, y1, x2, y2 = map(int, edge)
        cv2.line(contour_image, (x1, y1), (x2, y2), 255, 1)  # Draw refined edges

    contour_points = []
    for i in range(len(refined_corners) - 1):
        interpolated_points = interpolate_points(refined_corners[i], refined_corners[i + 1])
        contour_points += interpolated_points
    contour_points += interpolate_points(refined_corners[-1], refined_corners[0])
    return contour_image, contour_points


def interpolate_points(p1, p2):
    """
    Interpolate points between two given points (p1, p2) using linear interpolation.

    Args:
        p1 (tuple): Start point (x1, y1).
        p2 (tuple): End point (x2, y2).

    Returns:
        list: List of interpolated points between p1 and p2.
    """
    x1, y1 = p1
    x2, y2 = p2
    num_points = int(np.hypot(x2 - x1, y2 - y1))  # Number of points based on distance
    x_values = np.linspace(x1, x2, num_points)
    y_values = np.linspace(y1, y2, num_points)
    return [(int(round(x)), int(round(y))) for x, y in zip(x_values, y_values)]


def draw_polygon(image, points):
    """
    Draw the current polygon points on an image and annotate with area.

    Args:
        image (np.ndarray): Image to draw on.
        points (list): List of points in the polygon.
        area (float): Area of the polygon.

    Returns:
        np.ndarray: Updated image with the polygon and area drawn.
    """
    # Draw all points in the current list
    for i in range(len(points)):
        # Convert the point to tuple if it isn't
        point = tuple(map(int, points[i]))
        cv2.circle(image, point, radius=2, color=255, thickness=-1)

    # Draw the polygon lines
    for i in range(len(points) - 1):
        point_i = tuple(map(int, points[i]))
        point_i_1 = tuple(map(int, points[i + 1]))
        cv2.line(image, point_i, point_i_1, color=255, thickness=1)
    return image


def find_edges_from_contour(contour, area_threshold, neighbour_distance=2.0):
    """
    Find edges from the contour based on the algorithm provided.

    Args:
        contour (np.ndarray): Contour points as a NumPy array of shape (N, 1, 2).
        area_threshold (float): Threshold for the area enclosed by points to stop extending the list.

    Returns:
        list: List of edges, where each edge is a list of points forming a closed loop.
    """

    def calculate_area(points):
        """Calculate the area of a polygon formed by a list of points."""
        if len(points) < 3:
            return 0  # Not a polygon
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    contour_points = contour  # Flatten the contour to a list of points
    remaining_points = set(tuple(pt) for pt in contour_points)
    edges = []

    while remaining_points:
        current_list = []
        # Randomly pick a starting point
        start_point = remaining_points.pop()
        current_list.append(start_point)

        # Initialize direction as None
        direction = None
        # Initialize blank image
        image = np.zeros((64, 64), dtype=np.uint8)
        area = 0
        while True:
            # If there's only one point, choose any neighbor to establish direction
            if len(current_list) == 1:
                direction = np.array([1, 0])  # Arbitrary initial direction
            elif len(current_list) > 1:
                # Update direction based on the last two points
                direction = np.array(current_list[-1]) - np.array(current_list[-2])

            # Find the closest point to the current point
            distances = [
                (pt, np.linalg.norm(np.array(pt) - np.array(current_list[-1])))
                for pt in remaining_points
            ]
            distances.sort(key=lambda x: x[1])  # Sort by distance
            closest_point = distances[0][0] if distances else None

            if closest_point is None:
                break

            # Add interpolated points between the last point and the closest point
            interpolated_points = interpolate_points(current_list[-1], closest_point)
            current_list.extend(interpolated_points[1:])  # Add all except the first (it's already in the list)

            # Add the closest point to the list and remove it from the remaining points

            current_list.append(closest_point)
            remaining_points.discard(closest_point)

            # Calculate the area enclosed by the points in the list
            area = calculate_area(current_list)
            if area > area_threshold:
                break

        # Save the edge and start a new one
        edges.append(current_list)
        # Draw the current polygon and print the area
        image = draw_polygon(image, current_list)
        van(image)
    return edges


def calculate_area(points):
    """Calculate the area of a polygon formed by a list of points."""
    if len(points) < 3:
        return 0  # Not a polygon
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def can_merge_edges(edge1, edge2, distance_threshold=5):
    """
    Check if two edges can be merged based on their endpoints' proximity.

    Args:
        edge1 (list): First edge as a list of points.
        edge2 (list): Second edge as a list of points.
        distance_threshold (float): Maximum distance for endpoints to be considered close.

    Returns:
        bool: True if the edges can be merged, False otherwise.
    """
    # Check proximity of endpoints
    return (
            np.linalg.norm(np.array(edge1[-1]) - np.array(edge2[0])) <= distance_threshold or
            np.linalg.norm(np.array(edge2[-1]) - np.array(edge1[0])) <= distance_threshold or
            np.linalg.norm(np.array(edge2[-1]) - np.array(edge1[-1])) <= distance_threshold or
            np.linalg.norm(np.array(edge2[0]) - np.array(edge1[0])) <= distance_threshold
    )


def merge_edges(edge1, edge2):
    """
    Merge two edges into a single edge by combining their points.

    Args:
        edge1 (list): First edge as a list of points.
        edge2 (list): Second edge as a list of points.

    Returns:
        list: Merged edge.
    """
    # Ensure edges are connected at endpoints
    if edge1[-1] == edge2[0]:
        return edge1 + edge2[1:]
    elif edge1[0] == edge2[-1]:
        return edge2 + edge1[1:]
    elif edge1[-1] == edge2[-1]:
        return edge1 + edge2[::-1][1:]
    elif edge1[0] == edge2[0]:
        return edge2[::-1] + edge1[1:]
    return edge1 + edge2  # Fallback for disconnected edges


def merge_edges_with_area_constraint(edges, area_threshold):
    """
    Merge edges iteratively if the area after merging is smaller than the threshold.

    Args:
        edges (list): List of edges, where each edge is a list of connected points.
        area_threshold (float): Maximum allowed area for merging.

    Returns:
        list: List of merged edges.
    """
    merged = True
    while merged:
        merged = False
        new_edges = []
        used_edges = set()

        # Try all pairs of edges
        for edge1, edge2 in combinations(edges, 2):
            if edge1 in used_edges or edge2 in used_edges:
                continue

            # Merge the two edges and calculate the area
            merged_edge = merge_edges(edge1, edge2)
            merged_area = calculate_area(merged_edge)

            if merged_area <= area_threshold:
                # Merge the edges if the area is within the threshold
                new_edges.append(merged_edge)
                used_edges.add(edge1)
                used_edges.add(edge2)
                merged = True
                break  # Restart the process after merging

        # Add remaining edges that were not merged
        edges = [edge for edge in edges if edge not in used_edges] + new_edges

    # Draw the current polygon and print the area
    for edge in edges:
        image = np.zeros((64, 64), dtype=np.uint8)
        draw_polygon(image, edge)
        van(image)

    return edges


def find_similar_patches(test_image, dataset_patches, vertex_labels, shape_labels, patch_size=32, threshold=1000):
    """
    Find patches in the test image that are similar to those in the dataset.

    Args:
        test_image (np.ndarray): Test image as a NumPy array (grayscale).
        dataset_patches (list): List of patches from the dataset as NumPy arrays.
        vertex_labels (list): List of labels corresponding to the dataset patches.
        patch_size (int): Size of the patches to extract (default is 32x32).
        threshold (float): Threshold for similarity measure (lower is more similar).

    Returns:
        list: List of matched patches' labels and their positions in the test image.
    """
    matches = []
    half_size = patch_size // 2
    resize_to = 32
    # Crop the image to the valid area
    cropped_image, (x_offset, y_offset, _, _) = crop_to_valid_area(test_image, 0)
    if cropped_image.shape[0] > resize_to:
        # Resize the image using INTER_NEAREST interpolation to minimize smoothing
        resized_image = cv2.resize(cropped_image.astype(np.uint8), (resize_to, resize_to),
                                   interpolation=cv2.INTER_NEAREST)

        # Ensure the resized image is binary (0 or 1)
        resized_image = (resized_image > 0.5).astype(np.uint8)
    else:
        resized_image = cropped_image

    # Pad the test image to handle boundary cases
    padded_image = np.pad(resized_image, ((half_size, half_size), (half_size, half_size)), mode='constant',
                          constant_values=0)

    contour_img, contour_points = find_contour(padded_image)
    edges = find_edges_from_contour(contour_points, area_threshold=2, neighbour_distance=1)
    merged_edges = merge_edges_with_area_constraint([tuple(edge) for edge in edges], area_threshold=20)

    # Pre-compute flattened dataset patches for faster distance computation
    dataset_flat = np.array([p.flatten() for p in dataset_patches])

    dists = []
    shapes = []
    # Calculate scale factors to map positions back to the original image
    scale_x = cropped_image.shape[1] / resize_to
    scale_y = cropped_image.shape[0] / resize_to

    # Iterate over the resized image's pixels for patch extraction
    for y in range(half_size, resize_to + half_size):
        for x in range(half_size, resize_to + half_size):
            # Extract the patch
            patch = padded_image[y - half_size:y + half_size, x - half_size:x + half_size]
            # connected_image = connect_isolated_areas(patch)
            patch = patch.flatten()
            # Compute distances to all dataset patches
            distances = cdist([patch], dataset_flat, metric='sqeuclidean')[0]

            # Find the most similar patch in the dataset
            min_distance = np.min(distances)
            dists.append(min_distance)
            shapes.append(shape_labels[np.argmin(distances)])
            best_match_idx = np.argmin(distances)
            vertex_label = vertex_labels[best_match_idx]

            # Map the position back to the original image coordinates
            original_x = int((x - half_size) * scale_x + x_offset)
            original_y = int((y - half_size) * scale_y + y_offset)

            shape_label = shape_labels[best_match_idx]
            matches.append(
                {"vertex_label": vertex_label, "position": (original_x, original_y),  # Original image coordinates
                 "shape_label": shape_label,
                 "distance": min_distance})
    dists = torch.tensor(dists)
    dists_sorted, indices = torch.sort(dists)
    labels_sorted = torch.tensor(shapes)[indices]
    matches_sorted = [matches[i] for i in indices if matches[i]["shape_label"] == matches[indices[0]]["shape_label"]]

    return matches_sorted


def normalize_direction_vector(direction_vector):
    """
    Normalize a direction vector so that each (dx, dy) becomes a unit vector.

    Args:
        direction_vector (list): A list of (dx, dy) tuples.

    Returns:
        list: A normalized direction vector with each (dx, dy) as a unit vector.
    """
    normalized_vector = []
    for dx, dy in direction_vector:
        magnitude = np.sqrt(dx ** 2 + dy ** 2)
        if magnitude > 0:
            normalized_vector.append((dx / magnitude, dy / magnitude))
        else:
            normalized_vector.append((0, 0))  # Handle zero-magnitude case
    return normalized_vector


def find_contours(input_array):
    # Find contours with all points along the edges
    contours, _ = cv2.findContours(input_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Extract contours as Nx2 arrays of pixel positions
    contour_coordinates = []
    for contour in contours:
        # Convert (x, y) to (row, col) matching the original NumPy array indexing
        coords = contour.squeeze(axis=1)  # Convert (N, 1, 2) to (N, 2)
        coords[:, [0, 1]] = coords[:, [1, 0]]  # Swap columns to match NumPy indexing
        contour_coordinates.append(coords)

    contour_points = [point[:, [1, 0]] for point in contour_coordinates]

    all_points = np.concatenate(contour_points)

    contour_img = np.zeros((input_array.shape[0], input_array.shape[0], 3))
    from src import bk
    for i in range(len(all_points)):
        pos = all_points[i]
        contour_img[pos[1], pos[0]] = [255, 255, 255]
    van(contour_img)
    return contour_points


def calculate_dvs(contour_points):
    dvs = []
    for contour_list in contour_points:
        direction_vector = data_utils.contour_to_direction_vector(contour_list)
        direction_vector = torch.tensor(direction_vector)
        dvs.append(direction_vector)
    return dvs


def calculate_circular_variance(angles, wrap_size=20):
    # Convert degrees to radians
    angles_rad = angles * math.pi / 180
    # Wrap the data for circular variance calculation
    wrapped_radians = torch.cat([angles_rad, angles_rad[:wrap_size - 1]])  # Wrap around

    # Function to calculate circular variance
    def circular_variance(values):
        sin_vals = torch.sin(values)
        cos_vals = torch.cos(values)
        mean_cos = torch.mean(cos_vals)
        mean_sin = torch.mean(sin_vals)
        R = torch.sqrt(mean_cos ** 2 + mean_sin ** 2)
        return 1 - R

    # Compute circular variance for each window
    variances = []
    for i in range(len(angles_rad)):
        window = wrapped_radians[i:i + wrap_size]  # Get a 5-value window
        variance = circular_variance(window)
        variances.append(variance)

    # Convert variances to a tensor
    variances = torch.tensor(variances)
    return variances


def extract_line_curves(angles, seg_var_th = 1e-8):
    """
    Segments a series of angles into increasing, decreasing, or constant segments by checking the trends of future angles.

    Args:
        angles (array-like): Input angles in degrees.
        tolerance (float): Fluctuation tolerance to consider as 'constant' (in degrees).
        future_steps (tuple): Steps ahead to check (e.g., 5th and 10th angles).

    Returns:
        list of dict: Each dict contains the start index, end index, and segment type ('increasing', 'decreasing', 'constant').
    """

    circular_variances = calculate_circular_variance(angles)
    # chart_utils.show_line_chart(circular_variances, "circular_variances")

    all_seg = []
    check_list = []
    current_seg = []
    segment_labels = []
    future_step = 5
    # Find the first index where the smoothed trend stops decreasing
    for i in range(len(angles) - future_step):
        # post_variance = torch.var_mean(angles[i:i + future_step])[0]
        # if i < future_step:
        #     prev_variance = torch.var_mean(torch.cat((angles[-(future_step - i):], angles[:i])))[0]
        # else:
        #     prev_variance = torch.var_mean(angles[i - future_step:i])[0]
        if circular_variances[i] < 0.1:
            current_seg.append(i)
            check_list.append(i)
        else:
            if len(current_seg) > 10:
                seg_var = torch.var(circular_variances[current_seg[10:-10]])
                if seg_var < seg_var_th:
                    segment_labels.append("line")
                else:
                    segment_labels.append("circle")
                # chart_utils.show_line_chart(circular_variances[current_seg], "seg_var")
                # chart_utils.show_line_chart(angles[current_seg], "angles")
                # print("seg_var", seg_var)
                # print("")
                all_seg.append(current_seg)
            current_seg = []
            current_seg.append(i)
            check_list.append(i)
    if len(current_seg) > 10:
        all_seg.append(current_seg)
        seg_var = torch.var(circular_variances[current_seg[10:-10]])
        if seg_var < seg_var_th:
            segment_labels.append("line")
        else:
            segment_labels.append("circle")
    # Visual Segments
    # chart_utils.visual_multiple_segments(all_seg, angles)
    all_seg = [seg for seg in all_seg if len(seg) > 10]
    if (len(angles) - all_seg[-1][-1]) < 10 and all_seg[0][0] < 10 and segment_labels[0] == segment_labels[-1]:
        merged_seg = all_seg[-1] + all_seg[0]
        all_seg = all_seg[1:-1] + [merged_seg]
        segment_labels = segment_labels[1:-1] + [segment_labels[0]]
    if len(all_seg) != len(segment_labels):
        print("")
    return all_seg, segment_labels


def recall_match(args, bk_shapes, img):
    bw_img = np.array(Image.fromarray(img.numpy().astype('uint8')).convert("L"))
    bw_img[bw_img == 211] = 0
    bw_img[bw_img > 0] = 1

    # bw_img = resize_img(bw_img, 64)
    contour_points = find_contours(bw_img)

    dvs = calculate_dvs(contour_points)
    all_segments = []
    all_labels = []
    # find out the segments, and their labels (curve or line)
    for dv in dvs:
        segments, seg_labels = extract_line_curves(dv, seg_var_th=1e-8)
        all_segments.append(segments)
        all_labels.append(seg_labels)

    # visualize the segments on the original contour image
    chart_utils.visual_labeled_contours(bw_img.shape[0], all_segments, contour_points, all_labels)

    # Find similar patches
    dataset_patches = torch.cat([bk_shapes[i]["fm_repo"] for i in range(len(bk_shapes))], dim=0)
    dataset_patches[dataset_patches > 0] = 1
    vertex_labels = [bk_shapes[i]["labels"] for i in range(len(bk_shapes))]
    vertex_labels = torch.cat([vertex_labels[i] for i in range(len(vertex_labels))])
    shape_labels = [[i] * len(bk_shapes[i]["fm_repo"]) for i in range(len(bk_shapes))]
    shape_labels = list(itertools.chain.from_iterable(shape_labels))

    th = 0
    matches = find_similar_patches(bw_img, dataset_patches, vertex_labels, shape_labels, patch_size=32,
                                   threshold=10)
    while len(matches) == 0:
        th += 1
        matches = find_similar_patches(bw_img, dataset_patches, vertex_labels, shape_labels, patch_size=32,
                                       threshold=th)

    most_frequent_matches = extract_most_frequent_label_matches(matches)
    match_shape_id = most_frequent_matches[0]["shape_label"] + 1

    group = gestalt_group.Group(id=0, name=match_shape_id, input_signal=segment, onside_signal=None,
                                # memory_signal=group_data['recalled_bw_img'],
                                parents=None, coverage=None, color=seg_color)

    onside_shapes = []
    onside_percents = torch.zeros(len(bk_shapes))
    cropped_data_all = []
    kernels_all = []
    for b_i, bk_shape in enumerate(bk_shapes):
        input_fms, cropped_data = models.img2fm(img, bk_shape["kernels"].float())
        mem_fms, _ = recall_fms(input_fms, bk_shape["fm_repo"], reshape=args.obj_fm_size)
        # reasoning recalled fms to group
        group_data = reason.reason_fms(input_fms, mem_fms, reshape=args.obj_fm_size)
        onside_shapes.append(group_data["onside"])
        onside_percents[b_i] = group_data["onside_percent"]
        cropped_data_all.append(cropped_data)
        kernels_all.append(bk_shape["kernels"])
    best_shape = onside_percents.argmax()
    best_cropped_data = cropped_data_all[onside_percents.argmax()]
    onside_pixels = onside_shapes[best_shape].squeeze()
    best_shape = best_shape + 1
    best_kernel = kernels_all[onside_percents.argmax()]
    return onside_pixels, best_shape, best_cropped_data, best_kernel
