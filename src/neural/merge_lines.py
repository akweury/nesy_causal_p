# Created by jing at 28.01.25
import numpy as np


def lines_direction_vector(line_dict):
    """
    Compute an approximate direction vector for the line_dict
    using the first and last points in 'points'.
    Returns a normalized vector (vx, vy).
    """
    pts = line_dict["points"]
    if len(pts) < 2:
        return None
    x1, y1 = pts[0]
    x2, y2 = pts[-1]
    vec = np.array([x2 - x1, y2 - y1], dtype=float)
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-12)


def lines_are_collinear(lineA, lineB, angle_threshold=1e-2):
    """
    Check if two lines' direction vectors are (approximately) parallel or anti-parallel.
    Use dot product with a tolerance on (1 - |dot|).
    """
    vA = lines_direction_vector(lineA)
    vB = lines_direction_vector(lineB)
    if vA is None or vB is None:
        return False

    # dot ~ ±1 => same or opposite direction
    dot = abs(np.dot(vA, vB))  # [0..1], 1 => same/opp direction
    # if 1 - dot < angle_threshold => nearly collinear
    if (1.0 - dot) < angle_threshold:
        return True
    return False


def point_to_line_distance(pt, line_pt1, line_pt2):
    """
    Perpendicular distance from point `pt` to the infinite line
    defined by (line_pt1, line_pt2).
    """
    (x1, y1) = line_pt1
    (x2, y2) = line_pt2
    (px, py) = pt
    numerator = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    denom = np.hypot(y2 - y1, x2 - x1)
    return numerator / (denom + 1e-12)


def lines_on_same_infinite_line(lineA, lineB, distance_threshold=1e-1):
    """
    Check that all points of lineB lie close (<= distance_threshold)
    to the infinite line defined by lineA, and vice versa.
    This ensures they share the same infinite line, not just being parallel.
    """
    ptsA = lineA["points"]
    ptsB = lineB["points"]

    # Let A's infinite line be from first to last in A
    A1 = ptsA[0]
    A2 = ptsA[-1]

    # Check every point in B is within distance_threshold of line A
    for pb in ptsB:
        dist_pb = point_to_line_distance(pb, A1, A2)
        if dist_pb > distance_threshold:
            return False

    # (Optionally) also check every point in A is near line B
    B1 = ptsB[0]
    B2 = ptsB[-1]
    for pa in ptsA:
        dist_pa = point_to_line_distance(pa, B1, B2)
        if dist_pa > distance_threshold:
            return False

    return True


def are_collinear_and_on_same_line(lineA, lineB, angle_threshold=1e-2, distance_threshold=1e-1):
    """
    Combine the checks:
      1) lines_are_collinear => direction vectors ~ parallel
      2) lines_on_same_infinite_line => they lie on the same infinite line
    """
    if not lines_are_collinear(lineA, lineB, angle_threshold):
        return False
    if not lines_on_same_infinite_line(lineA, lineB, distance_threshold):
        return False
    return True


def lines_overlap_or_connected(lineA, lineB, connect_threshold=1e-1):
    """
    (Optional) Check if lineA and lineB are actually near/overlapping in 1D sense
    on that infinite line.
    - For a robust solution, you might project all endpoints onto the direction vector
      and check intervals for overlap.
    - Here, we do a simple 'endpoint distance' check to see if they are close.
    """
    ptsA = lineA["points"]
    ptsB = lineB["points"]

    # Potentially relevant endpoints:
    Aends = [ptsA[0], ptsA[-1]]
    Bends = [ptsB[0], ptsB[-1]]

    def dist(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    # If any pair of endpoints is within connect_threshold,
    # we consider them connected or overlapping.
    for aend in Aends:
        for bend in Bends:
            if dist(aend, bend) <= connect_threshold:
                return True
    return False


def merge_two_lines(lineA, lineB):
    """
    Merge two lines known to be on the same infinite line.
    Combine their point indices and coordinates. Sort the result by projecting
    onto the direction vector, or at least by x or y for demonstration.
    """
    new_indices = list(set(lineA["indices"] + lineB["indices"]))
    all_points = np.vstack((lineA["points"], lineB["points"]))

    # A simple approach: remove duplicates & sort by x->y
    # For a real approach, you'd sort by the parametric coordinate along the line's direction.
    all_points_unique = np.unique(all_points, axis=0)
    # For demonstration, just sort by x then y:
    all_points_sorted = all_points_unique[np.lexsort((all_points_unique[:, 1], all_points_unique[:, 0]))]

    return {
        "indices": new_indices,
        "points": all_points_sorted
    }


def merge_collinear_lines(line_list, angle_threshold=1e-2, distance_threshold=1e-2, connect_threshold=1e-1):
    """
    Repeatedly merge lines if:
      1) They are collinear (angle check)
      2) They lie on the same infinite line (distance check)
      3) They overlap or are connected in 1D
    Continue until no merges occur.
    """
    merged_something = True
    while merged_something and len(line_list) > 1:
        merged_something = False
        new_lines = []
        skip_set = set()

        for i in range(len(line_list)):
            if i in skip_set:
                continue

            current_line = line_list[i]
            for j in range(i + 1, len(line_list)):
                if j in skip_set:
                    continue

                line_j = line_list[j]
                # Check conditions
                if are_collinear_and_on_same_line(current_line, line_j,
                                                  angle_threshold=angle_threshold,
                                                  distance_threshold=distance_threshold):
                    if lines_overlap_or_connected(current_line, line_j, connect_threshold):
                        # Merge them
                        current_line = merge_two_lines(current_line, line_j)
                        skip_set.add(j)
                        merged_something = True

            new_lines.append(current_line)
            skip_set.add(i)

        # Remove any duplicates
        unique_new = []
        for ln in new_lines:
            # We'll say it's "duplicate" if we've already got the same set of points
            # or something similar. This can be refined further.
            if all(not np.array_equal(ln["points"], unq["points"]) for unq in unique_new):
                unique_new.append(ln)

        line_list = unique_new

    return line_list


###############################################################################
#                            DEMO / USAGE EXAMPLE                              #
###############################################################################
def demo_merge_lines():
    """
    Simple example to show how the merging might work.
    """
    # Suppose we have three lines that are all essentially on the same infinite line:
    #   lineA: points ~ x in [0,1]
    #   lineB: points ~ x in [1,2]
    #   lineC: points ~ x in [2,3]
    # Slightly shifted in y for demonstration.

    lineA = {
        "indices": [0, 1],
        "points": np.array([[0, 0], [1, 0.0001]])
    }
    lineB = {
        "indices": [2, 3],
        "points": np.array([[1, 0], [2, 0.0002]])
    }
    lineC = {
        "indices": [4, 5],
        "points": np.array([[2, 0], [3, 0.0003]])
    }
    line_list = [lineA, lineB, lineC]

    print("Before merge:")
    for i, ln in enumerate(line_list):
        print(f"Line {i}: {ln}")

    merged = merge_collinear_lines(line_list,
                                   angle_threshold=1e-2,
                                   distance_threshold=1e-2,
                                   connect_threshold=0.2)

    print("\nAfter merge:")
    for i, ln in enumerate(merged):
        print(f"Merged Line {i}: indices={ln['indices']}, points=\n{ln['points']}")

