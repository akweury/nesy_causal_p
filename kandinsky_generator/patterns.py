# Created by X at 04.11.23

import random
import math
import numpy as np

WIDTH = 640

MINSIZE = 10 * 5
MAXSIZE = 24 * 5


def generate_on_line_pair(min_obj_num, max_obj_num, color_dict):
    # MINSIZE = 10*5
    # MAXSIZE = 24*5
    MINSIZE = 3 * 5
    MAXSIZE = 6 * 5
    nshapes = random.randint(min_obj_num, max_obj_num)
    ncolors = len(list(color_dict.keys()))
    dx = math.cos(random.random() * math.pi * 2) * (WIDTH / 2 - MAXSIZE / 2)
    dy = math.sin(random.random() * math.pi * 2) * (WIDTH / 2 - MAXSIZE / 2)
    sx = WIDTH / 2 - dx
    sy = WIDTH / 2 + dy
    ex = WIDTH / 2 + dx
    ey = WIDTH / 2 - dy
    dx = ex - sx
    dy = ey - sy

    shapes = []
    is_reject = True
    while is_reject or len(shapes) < nshapes:
        shapes = []
        color_ids = []
        shape_ids = []
        for i in range(nshapes):
            r = random.random()
            cx = sx + r * dx
            cy = sy + r * dy
            size = random.randint(MINSIZE, MAXSIZE)
            obj_color = random.choice(list(color_dict.keys()))
            obj_shape = random.choice(["triangle", "circle", "square"])
            shape = {'shape': obj_shape, 'cx': cx, 'cy': cy, 'size': size, 'color': obj_color}
            shapes.append(shape)
            color_ids.append(obj_color)
            shape_ids.append(obj_shape)

            pairs = [(color_ids[i], shape_ids[i])
                     for i in range(len(color_ids))]
            if len(set(pairs)) < len(pairs):
                is_reject = False

    return shapes


def points_between(point1, point2, num_points):
    # Extract coordinates
    x1, y1 = point1
    x2, y2 = point2

    # Generate linearly spaced coordinates
    x_coords = np.linspace(x1, x2, num_points)
    y_coords = np.linspace(y1, y2, num_points)

    # Combine x and y coordinates
    points = np.vstack((x_coords, y_coords)).T
    return points.astype(int)


def generate_on_line(min_obj_num, max_obj_num, color_dict):
    MINSIZE = 3 * 5
    MAXSIZE = 6 * 5

    x_range = (0, WIDTH)
    y_range = (0, WIDTH)

    distance = 0
    p1, p2 = None, None
    while distance < WIDTH * 0.2 and p1 is None and p2 is None:
        p1 = np.array((np.random.uniform(*x_range), np.random.uniform(*y_range))).astype(np.int32)
        p2 = np.array((np.random.uniform(*x_range), np.random.uniform(*y_range))).astype(np.int32)
        distance = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    num_points = np.random.randint(min_obj_num, max_obj_num + 1)
    points = points_between(p1, p2, num_points)
    shapes = []
    color_ids = []
    shape_ids = []
    for i in range(num_points):
        size = random.randint(MINSIZE, MAXSIZE)
        obj_color = random.choice(list(color_dict.keys()))
        obj_shape = random.choice(["triangle", "circle", "square"])
        shape = {'shape': obj_shape, 'cx': points[i][0].item(), 'cy': points[i][1].item(), 'size': size, 'color': obj_color}
        shapes.append(shape)
        color_ids.append(obj_color)
        shape_ids.append(obj_shape)
    return shapes


def randomShapes(min_obj_num, max_obj_num, color_dict):
    MINSIZE = 10 * 5
    MAXSIZE = 24 * 5
    nshapes = random.randint(min_obj_num, max_obj_num)
    ncolors = len(list(color_dict.keys()))
    shapes = []
    for i in range(nshapes):
        cx = random.randint(MAXSIZE // 2, WIDTH - MAXSIZE // 2)
        cy = random.randint(MAXSIZE // 2, WIDTH - MAXSIZE // 2)
        obj_color = random.choice(list(color_dict.keys()))
        size = random.randint(MINSIZE, MAXSIZE)
        obj_shape = random.choice(["triangle", "circle", "square"])
        shape = {'shape': obj_shape, 'cx': cx, 'cy': cy, 'size': size, 'color': obj_color}

        shapes.append(shape)
    return shapes


def red_square(data_sign, min_obj_num, max_obj_num, color_dict):
    MINSIZE = 10 * 5
    MAXSIZE = 24 * 5
    nshapes = random.randint(min_obj_num, max_obj_num)
    ncolors = len(list(color_dict.keys()))
    shapes = []

    if data_sign == "true":
        cx = random.randint(MAXSIZE // 2, WIDTH - MAXSIZE // 2)
        cy = random.randint(MAXSIZE // 2, WIDTH - MAXSIZE // 2)
        size = random.randint(MINSIZE, MAXSIZE)
        red_square = {'shape': "square", 'cx': cx, 'cy': cy, 'size': size, 'color': "red"}
        shapes.append(red_square)
        i = 1
    else:
        i = 0

    while i < nshapes:
        cx = random.randint(MAXSIZE // 2, WIDTH - MAXSIZE // 2)
        cy = random.randint(MAXSIZE // 2, WIDTH - MAXSIZE // 2)
        obj_color = random.choice(list(color_dict.keys()))
        size = random.randint(MINSIZE, MAXSIZE)
        obj_shape = random.choice(["triangle", "circle", "square"])
        if not (obj_shape == "square" and obj_color == "red"):
            shape = {'shape': obj_shape, 'cx': cx, 'cy': cy, 'size': size, 'color': obj_color}
            shapes.append(shape)
            i += 1
    return shapes


def generate_pair_colors(data_sign, min_obj_num, max_obj_num, color_dict):
    MINSIZE = 5 * 5
    MAXSIZE = 10 * 5
    nshapes = random.randint(min_obj_num, max_obj_num)

    dx = math.cos(random.random() * math.pi * 2) * (WIDTH / 2 - MAXSIZE / 2)
    dy = math.sin(random.random() * math.pi * 2) * (WIDTH / 2 - MAXSIZE / 2)
    sx = WIDTH / 2 - dx
    sy = WIDTH / 2 + dy
    ex = WIDTH / 2 + dx
    ey = WIDTH / 2 - dy
    dx = ex - sx
    dy = ey - sy

    data = []
    inside_of_canvas = False

    # set rs pairs for negative patterns
    if data_sign == "true":
        pair_keys = [1] * (nshapes // 2)
    else:
        pair_keys = [random.randint(0, 1) for i in range(nshapes // 2)]
        while sum(pair_keys) == (nshapes // 2):
            pair_keys = [random.randint(0, 1) for i in range(nshapes // 2)]
    while not inside_of_canvas or len(data) < nshapes:
        data = []
        colors = []
        shapes = []
        for obj_i in range(nshapes):
            r = random.random()
            cx = sx + r * dx
            cy = sy + r * dy
            size = random.randint(MINSIZE, MAXSIZE)

            # color reflection symmetry
            if data_sign == "true":

                if obj_i >= (nshapes // 2) and (nshapes - obj_i) <= len(colors):
                    obj_color = colors[nshapes - 1 - obj_i]
                else:
                    obj_color = random.choice(list(color_dict.keys()))
            elif data_sign == "false":
                if obj_i >= (nshapes // 2) and (nshapes - obj_i) <= len(colors):
                    if pair_keys[nshapes - 1 - obj_i] == 1:
                        obj_color = colors[nshapes - 1 - obj_i]
                    else:
                        obj_color = random.choice(list(color_dict.keys()))
                        while obj_color == colors[nshapes - 1 - obj_i]:
                            obj_color = random.choice(list(color_dict.keys()))
                else:
                    obj_color = random.choice(list(color_dict.keys()))
            else:
                raise ValueError

            obj_shape = random.choice(["triangle", "circle", "square"])
            obj = {'shape': obj_shape, 'cx': cx, 'cy': cy, 'size': size, 'color': obj_color}
            data.append(obj)
            colors.append(obj_color)
            shapes.append(obj_shape)

        for obj in data:
            inside_of_canvas = True
            half_side_length = obj['size'] * 0.5
            if obj['cx'] + half_side_length > WIDTH or obj['cx'] - half_side_length < 0 or obj[
                'cy'] - half_side_length < 0 or obj['cy'] + half_side_length > WIDTH:
                inside_of_canvas = False
                break
    return data


def generate_reflection_symmetry(data_sign, min_obj_num, max_obj_num, color_dict, color_rs=False, shape_rs=False):
    MINSIZE = 5 * 5
    MAXSIZE = 15 * 5
    nshapes = random.randint(min_obj_num, max_obj_num)
    shift_size = 100

    # position setting
    sx = random.random() * WIDTH
    # left to right
    if sx < WIDTH / 5:
        sy = random.random() * WIDTH
        dx = shift_size

        # top to bottom
        if sy < WIDTH / 2:
            dy = shift_size / 2
        else:
            dy = -shift_size / 2

    # top to bottom
    else:
        sy = random.random() * WIDTH / 5
        dy = shift_size
        # top to bottom
        if sx > WIDTH / 2:
            dx = -shift_size / 2
        else:
            dx = shift_size / 2

    data = []
    inside_of_canvas = False

    # set rs pairs for negative patterns
    if data_sign == "true":
        pair_keys = [1] * (nshapes // 2)
    else:
        pair_keys = [random.randint(0, 1) for i in range(nshapes // 2)]
        while sum(pair_keys) == (nshapes // 2):
            pair_keys = [random.randint(0, 1) for i in range(nshapes // 2)]

    while not inside_of_canvas or len(data) < nshapes:
        data = []
        colors = []
        shapes = []
        for obj_i in range(nshapes):
            r = random.random()
            if obj_i == 0:
                cx = sx + r * dx
                cy = sy + r * dy
            else:
                cx = data[obj_i - 1]['cx'] + r * dx
                cy = data[obj_i - 1]['cy'] + r * dy

            size = random.randint(MINSIZE, MAXSIZE)

            # color reflection symmetry
            if color_rs and data_sign == "true":
                if obj_i >= (nshapes // 2) and (nshapes - obj_i) <= len(colors):
                    obj_color = colors[nshapes - 1 - obj_i]
                else:
                    obj_color = random.choice(list(color_dict.keys()))
            elif color_rs and data_sign == "false":
                if obj_i >= (nshapes // 2) and (nshapes - obj_i) <= len(colors):
                    if pair_keys[nshapes - 1 - obj_i] == 1:
                        obj_color = colors[nshapes - 1 - obj_i]
                    else:
                        obj_color = random.choice(list(color_dict.keys()))
                        while obj_color == colors[nshapes - 1 - obj_i]:
                            obj_color = random.choice(list(color_dict.keys()))
                else:
                    obj_color = random.choice(list(color_dict.keys()))
            else:
                obj_color = random.choice(list(color_dict.keys()))

            # shape reflection symmetry
            if shape_rs and data_sign == "true":
                if obj_i >= (nshapes // 2) and (nshapes - obj_i) <= len(shapes):
                    obj_shape = shapes[nshapes - 1 - obj_i]
                else:
                    obj_shape = random.choice(["triangle", "circle", "square"])
            elif shape_rs and data_sign == "false":
                if obj_i >= (nshapes // 2) and (nshapes - obj_i) <= len(shapes):
                    if pair_keys[nshapes - 1 - obj_i] == 1:
                        obj_shape = shapes[nshapes - 1 - obj_i]
                    else:
                        obj_shape = random.choice(["triangle", "circle", "square"])
                        while obj_shape == shapes[nshapes - 1 - obj_i]:
                            obj_shape = random.choice(["triangle", "circle", "square"])
                else:
                    obj_shape = random.choice(["triangle", "circle", "square"])
            else:
                obj_shape = random.choice(["triangle", "circle", "square"])

            obj = {'shape': obj_shape, 'cx': cx, 'cy': cy, 'size': size, 'color': obj_color}

            data.append(obj)
            colors.append(obj_color)
            shapes.append(obj_shape)
        for obj in data:
            inside_of_canvas = True
            half_side_length = obj['size'] * 0.5
            if obj['cx'] + half_side_length > WIDTH or obj['cx'] - half_side_length < 0 or obj[
                'cy'] - half_side_length < 0 or obj['cy'] + half_side_length > WIDTH:
                inside_of_canvas = False
                break
    return data
