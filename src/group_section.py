# Created by jing at 09.07.24

import numpy as np


def find_connected_components(matrix):
    rows, cols = matrix.shape
    visited = np.zeros_like(matrix, dtype=bool)
    components = []

    def dfs(x, y, value):
        stack = [(x, y)]
        component = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

        while stack:
            cx, cy = stack.pop()
            if visited[cx, cy]:
                continue
            visited[cx, cy] = True
            component.append((cx, cy))

            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny]:
                    if matrix[nx, ny] == value:
                        stack.append((nx, ny))

        return component

    for i in range(rows):
        for j in range(cols):
            if not visited[i, j]:
                component = dfs(i, j, matrix[i, j])
                components.append(component)

    return components


def is_rectangle(component):
    x_coords = [pos[0] for pos in component]
    y_coords = [pos[1] for pos in component]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            if (x, y) not in component:
                return False

    return True


def is_grid_splitter(component, matrix_shape):
    rows, cols = matrix_shape
    x_coords = [pos[1] for pos in component]
    y_coords = [pos[0] for pos in component]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # check row lines, if the first and second tile are in the component, then the whole row should in the component
    for r_i in range(rows):
        if (r_i, 0) in component and (r_i, 1) in component:
            for c_i in range(2, cols):
                if (r_i, c_i) not in component:
                    return False

    # check col lines
    for c_i in range(cols):
        if (0, c_i) in component and (1, c_i) in component:
            for r_i in range(2, rows):
                if (r_i, c_i) not in component:
                    return False

    return True


def grid_splitting(matrix):
    components = find_connected_components(matrix)
    splitter_index = None
    for c_i in range(len(components)):
        if is_grid_splitter(components[c_i], matrix.shape):
            splitter_index = c_i
            break

    rect_regions = []
    for c_i in range(len(components)):
        if c_i != splitter_index:
            rect_regions.append(components[c_i])
    return rect_regions


def bar_splitting(matrix):
    rows, cols = matrix.shape
    components = find_connected_components(matrix)

    for component in components:
        if is_rectangle(component):
            x_coords = [pos[0] for pos in component]
            y_coords = [pos[1] for pos in component]
            if min(x_coords) == 0 and max(x_coords) == rows - 1:
                left_part = [(x, y) for x in range(rows) for y in range(cols) if y < min(y_coords)]
                right_part = [(x, y) for x in range(rows) for y in range(cols) if y > max(y_coords)]
                return [left_part, right_part]
            if min(y_coords) == 0 and max(y_coords) == cols - 1:
                top_part = [(x, y) for x in range(rows) for y in range(cols) if x < min(x_coords)]
                bottom_part = [(x, y) for x in range(rows) for y in range(cols) if x > max(x_coords)]
                return [top_part, bottom_part]

    return None
def connection_splitting(matrix):
    rows, cols = matrix.shape
    components = find_connected_components(matrix)

    for component in components:
        if is_rectangle(component):
            x_coords = [pos[0] for pos in component]
            y_coords = [pos[1] for pos in component]
            if min(x_coords) == 0 and max(x_coords) == rows - 1:
                left_part = [(x, y) for x in range(rows) for y in range(cols) if y < min(y_coords)]
                right_part = [(x, y) for x in range(rows) for y in range(cols) if y > max(y_coords)]
                return [left_part, right_part]
            if min(y_coords) == 0 and max(y_coords) == cols - 1:
                top_part = [(x, y) for x in range(rows) for y in range(cols) if x < min(x_coords)]
                bottom_part = [(x, y) for x in range(rows) for y in range(cols) if x > max(x_coords)]
                return [top_part, bottom_part]

    return None