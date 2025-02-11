# Created by x at 29.01.25
import itertools
from collections import defaultdict
import numpy as np
from collections import Counter
from math import sqrt


def are_collinear(p1, p2, p3, th=1e-3):
    """Check if three points are collinear using the determinant method."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return np.abs(((y2 - y1) * (x3 - x1) - (y3 - y1) * (x2 - x1))) < th


def detect_lines(points, min_points=4):
    """Find all lines consisting of at least min_points from a set of points."""
    points = [tuple(point) for point in points]
    indexed_points = list(enumerate(points))  # [(index, (x, y)), ...]
    n = len(indexed_points)
    lines = []
    seen_lines = {}

    for (i1, p1), (i2, p2) in itertools.combinations(indexed_points, 2):
        line_points = {(i1, p1), (i2, p2)}

        for i3, p3 in indexed_points:
            if (i3, p3) not in line_points and are_collinear(p1, p2, p3, th=1e-3):
                line_points.add((i3, p3))

        if len(line_points) >= min_points:
            # Create a unique identifier for the line
            line_tuple = tuple(sorted(line_points))
            if line_tuple not in seen_lines:
                seen_lines[line_tuple] = True
                # Store as dictionary with indices and points
                lines.append({
                    "indices": [idx for idx, _ in line_points],
                    "points": [point for _, point in line_points]
                })

    return lines


def euclidean_distance(p1, p2):
    """Compute Euclidean distance between two 2D points."""
    return sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def best_fit_line_projection(points):
    """
    Projects points onto a best-fit line using linear regression.

    :param points: List of (x, y) tuples
    :return: Sorted points along the best-fit line with their original indices
    """
    points = np.array(points)
    x, y = points[:, 0], points[:, 1]

    # Fit a line (y = mx + b) using least squares
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]  # m: slope, b: intercept

    # Compute projection scalar t = (x + my) / (1 + m^2)
    t_values = (x + m * y) / (1 + m ** 2)

    # Sort by projected position along the best-fit line
    sorted_indices = np.argsort(t_values)
    sorted_points = [tuple(points[i]) for i in sorted_indices]

    return sorted_points, sorted_indices.tolist()


def find_best_starting_point(sorted_points, most_common_distance, tolerance):
    """
    Finds the best starting point to maximize equally spaced points.

    :param sorted_points: List of sorted 2D points
    :param most_common_distance: The most frequently occurring distance
    :param tolerance: Allowed deviation for spacing
    :return: Index of the best starting point in sorted_points
    """
    best_start_index = 0
    max_kept_points = 0

    # Iterate over all points to see which gives the best sequence
    for start_index in range(len(sorted_points)):
        kept_count = 1  # Count how many points can be kept
        last_point = sorted_points[start_index]

        for i in range(start_index + 1, len(sorted_points)):
            distance = euclidean_distance(sorted_points[i], last_point)
            if (most_common_distance * (1 - tolerance)) <= distance <= (most_common_distance * (1 + tolerance)):
                kept_count += 1
                last_point = sorted_points[i]

        if kept_count > max_kept_points:
            max_kept_points = kept_count
            best_start_index = start_index

    return best_start_index


def filter_equally_spaced_2d_points(points, tolerance=0.05):
    """
    Filters points that are roughly equally spaced along their best-fit line.

    :param points: List of (x, y) tuples
    :param tolerance: Acceptable percentage deviation from the most common spacing (default 5%)
    :return: Filtered points and their original indices
    """
    if len(points) < 3:
        return points, list(range(len(points)))  # If less than 3 points, return as is.

    sorted_points, sorted_indices = best_fit_line_projection(points)

    # Compute distances between consecutive points
    distances = [euclidean_distance(sorted_points[i], sorted_points[i + 1]) for i in range(len(sorted_points) - 1)]

    # Find the most common distance (mode)
    most_common_distance, _ = Counter(round(d, 2) for d in distances).most_common(1)[0]
    # Find the best starting point
    best_start_index = find_best_starting_point(sorted_points, most_common_distance, tolerance)

    # Define tolerance range
    lower_bound = most_common_distance * (1 - tolerance)
    upper_bound = most_common_distance * (1 + tolerance)

    # Use this best starting point to filter points
    filtered_points = [sorted_points[best_start_index]]
    filtered_indices = [sorted_indices[best_start_index]]
    last_point = sorted_points[best_start_index]

    for i in range(best_start_index + 1, len(sorted_points)):
        distance = euclidean_distance(sorted_points[i], last_point)
        if (most_common_distance * (1 - tolerance)) <= distance <= (most_common_distance * (1 + tolerance)):
            filtered_points.append(sorted_points[i])
            filtered_indices.append(sorted_indices[i])
            last_point = sorted_points[i]

    return filtered_points, filtered_indices


def update_lines(lines, points_to_check):
    """Update each detected line with additional points if they are collinear with the line."""
    indexed_points_to_check = list(enumerate(points_to_check))  # Ensure unique indices

    for line in lines:
        base_indices = line["indices"]
        base_points = line["points"]

        # Take the first two points from the line as reference
        p1, p2 = base_points[:2]

        for i, p3 in indexed_points_to_check:
            if are_collinear(p1, p2, p3) and i not in base_indices:
                line["indices"].append(i)
                line["points"].append(tuple(p3))

        # remove the outlier in the line
        filtered, indices = filter_equally_spaced_2d_points(line["points"], tolerance=0.06)
        line["points"] = filtered
        line["indices"] = [line["indices"][i] for i in indices]
        # Sort after adding new points
        line["indices"], line["points"] = zip(*sorted(zip(line["indices"], line["points"]), key=lambda x: x[0]))
    # unique lines
    unique_lines = []
    unique_lines_indices = []
    for line in lines:
        if line["indices"] not in unique_lines_indices and len(line["indices"]) > 3:
            unique_lines.append(line)
            unique_lines_indices.append(line["indices"])

    return unique_lines


def best_fit_line(points):
    """
    Finds the best-fit line (y = mx + b) for the given points.

    :param points: List of (x, y) tuples
    :return: (slope, intercept)
    """
    points = np.array(points)
    x, y = points[:, 0], points[:, 1]

    # Fit a line y = mx + b using least squares
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]  # m: slope, b: intercept

    return m, b


def project_points_onto_line(points, m, b):
    """
    Projects points onto the best-fit line and finds the start and end points.

    :param points: List of (x, y) tuples
    :param m: Slope of the best-fit line
    :param b: Intercept of the best-fit line
    :return: (start_point, end_point)
    """
    points = np.array(points)
    x, y = points[:, 0], points[:, 1]

    # Compute projection scalar t = (x + my) / (1 + m^2)
    t_values = (x + m * y) / (1 + m ** 2)

    # Find min/max projected points
    min_index, max_index = np.argmin(t_values), np.argmax(t_values)
    start_point, end_point = points[min_index], points[max_index]

    return tuple(start_point), tuple(end_point)


def get_line_data(all_lines):
    """
    Given a set of points (Nx2 tuples) on a line, determine the slope, start point, and end point.

    Args:
    points (list of tuples): List of (x, y) coordinates.

    Returns:
    dict: A dictionary with 'slope', 'start_point', and 'end_point'.
    """
    line_data = []
    for line in all_lines:
        # Convert to numpy array for easy manipulation
        points = np.array(line["points"])

        m, b = best_fit_line(points)
        start_point, end_point = project_points_onto_line(points, m, b)

        # rounded_points = {(round(x, 2), round(y, 2)) for x, y in points}
        # Convert to a sorted list based on x first, then y
        # sorted_points = sorted(rounded_points, key=lambda p: (p[0], p[1]))

        # Determine start and end points
        # start_point = tuple(sorted_points[0])
        # end_point = tuple(sorted_points[-1])

        # Calculate slope (m = (y2 - y1) / (x2 - x1))
        if end_point[0] != start_point[0]:  # Avoid division by zero
            slope = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])
            if slope > 10:
                slope = 10
        else:
            slope = 10  # Undefined slope (vertical line)
        line_data.append([slope, start_point, end_point])
    return line_data
