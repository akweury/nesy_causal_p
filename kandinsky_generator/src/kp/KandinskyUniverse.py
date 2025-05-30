import math

import matplotlib
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageColor

from src import bk
from mbg import patch_preprocess

class kandinskyShape:
    def __init__(self, shape="", color="", x=0.5, y=0.5, size=0.5, line_width=1.0,
                 solid=False, start_angle=0, end_angle=math.pi*2 / 3,):
        self.shape = shape
        self.color = color
        self.x = x
        self.y = y
        self.size = size
        self.line_width = line_width
        self.solid = solid
        self.start_angle = start_angle
        self.end_angle = end_angle

    def __str__(self):
        return self.color + " " + self.shape + " (" + \
            str(self.size) + "," + str(self.x) + "," + str(self.y) + ")"


class SimpleUniverse:
    kandinsky_colors = ['red', 'yellow', 'blue']
    kandinsky_shapes = ['rectangle', 'circle', 'triangle']


# class ExtendedUniverse:
#     # still have to add drawing functions below
#     kandinsky_colors = ['red', 'yellow', 'blue', "green", "orange"]
#     kandinsky_shapes = ['rectangle', 'circle', 'triangle', "star"]


def rectangle(d, cx, cy, s, f):
    s = 0.6 * s
    d.rectangle(((cx - s / 2, cy - s / 2), (cx + s / 2, cy + s / 2)), fill=f)


def circle(d, cx, cy, s, f):
    # correct the size to  the same area as an rectangle
    s = 0.6 * math.sqrt(4 * s * s / math.pi)
    d.ellipse(((cx - s / 2, cy - s / 2), (cx + s / 2, cy + s / 2)), fill=f)


def triangle(d, cx, cy, s, f):
    r = math.radians(30)
    # correct the size to  the same area as an rectangle
    s = 0.6 * math.sqrt(4 * s * s / math.sqrt(3))
    s = math.sqrt(3) * s / 3
    dx = s * math.cos(r)
    dy = s * math.sin(r)
    d.polygon([(cx, cy - s), (cx + dx, cy + dy), (cx - dx, cy + dy)], fill=f)


def pac_man(d, cx, cy, s, f):
    # correct the size to  the same area as an rectangle
    s = 0.6 * math.sqrt(4 * s * s / math.pi)
    d.ellipse(((cx - s / 2, cy - s / 2), (cx + s / 2, cy + s / 2)), fill=f)


def diamond(d, cx, cy, size, f=True):
    """
    Draws a diamond on the given image.

    :param draw: ImageDraw object to draw on.
    :param cx: the x position of the diamond.
    :param cy: the y position of the diamond.
    :param size: Size of the diamond (distance from center to the points).
    :param color: Color of the diamond (RGB or RGBA).
    """

    # Calculate the four points of the diamond based on center and size
    diamond_points = [
        (cx, cy - size),  # Top point
        (cx + size, cy),  # Right point
        (cx, cy + size),  # Bottom point
        (cx - size, cy)  # Left point
    ]

    # Draw the diamond
    d.polygon(diamond_points, fill=f)


def heart(draw, cx, cy, size, color=True):
    """
    Draws a heart on the given image.

    :param draw: ImageDraw object to draw on.
    :param cx: X-coordinate of the center of the heart.
    :param cy: Y-coordinate of the center of the heart.
    :param size: Size of the heart.
    :param color: Color of the heart (RGB or RGBA).
    """
    # Calculate the width and height of the heart based on the size
    width = size
    height = size * 1.2  # Hearts are usually taller than wide

    # Define the points for the bottom triangle part of the heart
    triangle_points = [
        (cx, cy + height // 4),  # Bottom tip of the heart
        (cx - width // 2, cy),  # Left edge
        (cx + width // 2, cy)  # Right edge
    ]

    # Draw the bottom triangle
    draw.polygon(triangle_points, fill=color)

    # Draw the two circles at the top
    left_circle_bbox = [
        (cx - width // 2, cy - height // 4),  # Top-left corner of bounding box
        (cx, cy + height // 4)  # Bottom-right corner of bounding box
    ]

    right_circle_bbox = [
        (cx, cy - height // 4),  # Top-left corner of bounding box
        (cx + width // 2, cy + height // 4)  # Bottom-right corner of bounding box
    ]

    draw.ellipse(left_circle_bbox, fill=color)
    draw.ellipse(right_circle_bbox, fill=color)


def draw_star(image, center, size, color, thickness=-1):
    """
    Draws a star on the given image.

    :param image: NumPy array representing the image.
    :param center: Tuple (x, y) representing the center of the star.
    :param size: Size of the star (scale factor).
    :param color: Color of the star (BGR).
    :param thickness: Thickness of the star's outline. If -1, the star will be filled.
    """
    x, y = center

    # Star points relative to (0, 0) and then scaled by `size`
    star_points = np.array([
        [0, -size], [size * 0.2, -size * 0.2], [size, -size * 0.2],
        [size * 0.4, size * 0.2],
        [size * 0.6, size], [0, size * 0.5], [-size * 0.6, size],
        [-size * 0.4, size * 0.2],
        [-size, -size * 0.2], [-size * 0.2, -size * 0.2]
    ], np.int32)

    # Offset the star points by the (x, y) position of the center
    star_points[:, 0] += x
    star_points[:, 1] += y

    # Draw and fill the star
    cv2.polylines(image, [star_points], isClosed=True, color=color, thickness=2)
    if thickness == -1:
        cv2.fillPoly(image, [star_points], color=color)


def draw_pac_man(img, center_pos, obj_size, color, start_angle, end_angle):
    # Pac-Man parameters
    thickness = -1  # Filled shape
    mouth_angle = 60  # Total mouth opening angle in degrees

    # Calculate start and end angles.
    # OpenCV measures angles in degrees starting from the positive x-axis (to the right) and going clockwise.
    # To have the mouth centered to the right, we remove a wedge of "mouth_angle" centered at 0°.

    # Draw Pac-Man as a filled ellipse (which will draw a pie slice)
    cv2.ellipse(img, center_pos, (obj_size, obj_size), 0, start_angle, end_angle, color, thickness)


def draw_diamond(image, center, size, color, thickness=-1):
    """
    Draws a diamond on the given image.

    :param image: NumPy array representing the image.
    :param center: Tuple (x, y) representing the center of the diamond.
    :param size: Size of the diamond (distance from center to the points).
    :param color: Color of the diamond (BGR).
    :param thickness: Thickness of the outline. If -1, the diamond will be filled.
    """
    x, y = center

    # Define the four points of the diamond
    diamond_points = np.array([
        [x, y - size],  # Top point
        [x + size, y],  # Right point
        [x, y + size],  # Bottom point
        [x - size, y]  # Left point
    ], np.int32)

    diamond_points = diamond_points.reshape((-1, 1, 2))  # Reshape for polylines

    # Draw the diamond
    if thickness == -1:
        cv2.fillPoly(image, [diamond_points], color)
    else:
        cv2.polylines(image, [diamond_points], isClosed=True, color=color,
                      thickness=thickness)


def kandinskyFigureAsImagePIL(shapes, width=600, subsampling=4):
    image = Image.new("RGBA", (subsampling * width,
                               subsampling * width), (215, 215, 215, 255))
    d = ImageDraw.Draw(image)
    w = subsampling * width

    for s in shapes:
        globals()[s.shape](d, w * s.x, w * s.y, w * s.size, s.color)
    if subsampling > 1:
        image.thumbnail((width, width), Image.ANTIALIAS)

    return image


# def get_rgb_pastel(color):
#     # load clevr colormap
#     if color == "red":
#         return [173, 35, 35]
#     if color == "yellow":
#         return [255, 238, 51]
#     if color == "blue":
#         return [42, 75, 215]



def draw_circle(img, s, w):
    radius = 0.3 * w * s.size
    cx = round(w * s.x)
    cy = round(w * s.y)
    color = bk.color_matplotlib[s.color]
    thickness = -1 if s.solid else int(radius * s.line_width)
    cv2.circle(img, (cx, cy), round(radius), color, thickness)

def draw_triangle(img, s, w):
    angle = math.radians(30)
    size = 0.7 * math.sqrt(3) * w * s.size / 3
    dx = size * math.cos(angle)
    dy = size * math.sin(angle)

    cx, cy = round(w * s.x), round(w * s.y)
    p1 = (cx, round(cy - size))
    p2 = (round(cx + dx), round(cy + dy))
    p3 = (round(cx - dx), round(cy + dy))
    points = np.array([p1, p2, p3])

    color = bk.color_matplotlib[s.color]
    cv2.fillConvexPoly(img, points, color)

    if not s.solid:
        lwx = int(dx * s.line_width)
        lwy = int(dy * s.line_width)
        p1 = (cx, round(cy - size + size * s.line_width))
        p2 = (round(cx + dx - lwx), round(cy + dy - lwy))
        p3 = (round(cx - dx + lwx), round(cy + dy - lwy))
        top = np.array([p1, p2, p3])
        cv2.fillConvexPoly(img, top, bk.color_matplotlib["lightgray"])

def draw_rectangle(img, s, w):
    size = 0.3 * w * s.size
    color = bk.color_matplotlib[s.color]
    line_width = int(size * s.line_width)
    cx, cy = round(w * s.x), round(w * s.y)

    x1, y1 = round(cx - size), round(cy - size)
    x2, y2 = round(cx + size), round(cy + size)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

    if not s.solid:
        inner = [x1 + line_width, y1 + line_width, x2 - line_width, y2 - line_width]
        cv2.rectangle(img, (inner[0], inner[1]), (inner[2], inner[3]), bk.color_matplotlib["lightgray"], -1)

def kandinskyFigureAsImage(shapes, width=600, subsampling=1):
    w = subsampling * width
    img = np.full((w, w, 3), fill_value=bk.color_matplotlib["lightgray"], dtype=np.uint8)
    patch_inputs = []
    for s in shapes:
        obj_img = np.full((w, w, 3), fill_value=bk.color_matplotlib["lightgray"], dtype=np.uint8)
        shape = s.shape
        if shape == "circle":
            draw_circle(obj_img, s, w)
            draw_circle(img, s, w)
        elif shape == "triangle":
            draw_triangle(obj_img, s, w)
            draw_triangle(img, s, w)
        elif shape == "rectangle":
            draw_rectangle(obj_img, s, w)
            draw_rectangle(img, s, w)
        elif shape == "pac_man":
            center = (round(w * s.x), round(w * s.y))
            radius = round(0.3 * w * s.size)
            draw_pac_man(obj_img, center, radius, bk.color_matplotlib[s.color], s.start_angle, s.end_angle)
            draw_pac_man(img, center, radius, bk.color_matplotlib[s.color], s.start_angle, s.end_angle)
        elif shape == "star":
            size = 0.7 * math.sqrt(3) * w * s.size / 3
            draw_star(obj_img, (s.x, s.y), size, bk.color_matplotlib[s.color])
            draw_star(img, (s.x, s.y), size, bk.color_matplotlib[s.color])
        elif shape == "diamond":
            size = 0.7 * math.sqrt(3) * w * s.size / 3
            draw_diamond(obj_img, (w * s.x, w * s.y), size, bk.color_matplotlib[s.color])
            draw_diamond(img, (w * s.x, w * s.y), size, bk.color_matplotlib[s.color])
        else:
            raise ValueError(f"Unknown shape: {shape}")

        binary_np = patch_preprocess.rgb_to_binary(obj_img)
        patch_set = patch_preprocess.preprocess_image_to_patch_set(binary_np, contour_uniform=False)
        patch_set[0] = list(patch_set[0])
        patch_set[0][0] = patch_set[0][0].tolist()
        patch_set[0][1] = list(patch_set[0][1])
        patch_inputs.append(patch_set)

    img_resized = cv2.resize(img, (width, width), interpolation=cv2.INTER_AREA)
    return img_resized, patch_inputs


# def kandinskyFigureAsImage(shapes, width=600, subsampling=1):
#     w = subsampling * width
#     img = np.zeros((w, w, 3), np.uint8)
#     img[:, :] = bk.color_matplotlib["lightgray"]
#
#     for s in shapes:
#         # not sure if this is the right color for openCV
#         # rgbcolorvalue = ImageColor.getrgb(s.color)
#         # use pastel colors
#         rgbcolorvalue = bk.color_matplotlib[s.color]
#
#         if s.shape == "circle":
#             size = 0.5 * 0.6 * math.sqrt(4 * w * s.size * w * s.size / math.pi)
#             if s.solid:
#                 line_width = -1
#             else:
#                 line_width = int(size * s.line_width)
#             cx = round(w * s.x)
#             cy = round(w * s.y)
#             cv2.circle(img, (cx, cy), round(size), rgbcolorvalue,
#                        thickness=line_width)
#
#         elif s.shape == "triangle":
#             r = math.radians(30)
#             size = 0.7 * math.sqrt(3) * w * s.size / 3
#
#             # base shape
#             dx = size * math.cos(r)
#             dy = size * math.sin(r)
#             p1 = (round(w * s.x), round(w * s.y - size))
#             p2 = (round(w * s.x + dx), round(w * s.y + dy))
#             p3 = (round(w * s.x - dx), round(w * s.y + dy))
#             points = np.array([p1, p2, p3])
#             cv2.fillConvexPoly(img, points, rgbcolorvalue, 1)
#
#             # top shape
#             if not s.solid:
#                 line_x_width = int(dx * s.line_width)
#                 line_y_width = int(dy * s.line_width)
#                 p1 = (round(w * s.x),
#                       round(w * s.y - size + int(size * s.line_width)))  # top point
#                 p2 = (round(w * s.x + dx - line_x_width),
#                       round(w * s.y + dy - line_y_width))  # right point
#                 p3 = (round(w * s.x - dx + line_x_width),
#                       round(w * s.y + dy - line_y_width))  # left point
#                 points = np.array([p1, p2, p3])
#
#                 cv2.fillConvexPoly(img, points, bk.color_matplotlib["lightgray"], 1)
#
#         elif s.shape == "rectangle":
#             size = 0.5 * 0.6 * w * s.size
#             line_width = int(size * s.line_width)
#
#             # draw base rectangle
#             xs = round(w * s.x - size)
#             ys = round(w * s.y - size)
#             xe = round(w * s.x + size)
#             ye = round(w * s.y + size)
#             cv2.rectangle(img, (xs, ys), (xe, ye), rgbcolorvalue,
#                           thickness=-1)
#
#             # draw top rectangle
#             if not s.solid:
#                 xs = round(w * s.x - size + line_width)
#                 ys = round(w * s.y - size + line_width)
#                 xe = round(w * s.x + size - line_width)
#                 ye = round(w * s.y + size - line_width)
#                 cv2.rectangle(img, (xs, ys), (xe, ye),
#                               bk.color_matplotlib["lightgray"],
#                               thickness=-1)
#         elif s.shape == "pac_man":
#             cx = round(w * s.x)
#             cy = round(w * s.y)
#             size = round(0.5 * 0.6 * math.sqrt(4 * w * s.size * w * s.size / math.pi))
#
#             draw_pac_man(img, (cx, cy), size, rgbcolorvalue, s.start_angle, s.end_angle)
#         elif s.shape == "star":
#             size = 0.7 * math.sqrt(3) * w * s.size / 3
#             draw_star(img, (s.x, s.y), size, rgbcolorvalue)
#         elif s.shape == "diamond":
#             size = 0.7 * math.sqrt(3) * w * s.size / 3
#             draw_diamond(img, (w * s.x, w * s.y), size, rgbcolorvalue)
#         else:
#             raise ValueError("Unknown shape " + s.shape)
#
#     img_resampled = cv2.resize(
#         img, (width, width), interpolation=cv2.INTER_AREA)
#
#     image = Image.fromarray(img_resampled)
#
#     return image


def kandinskyFigureAsAnnotation(shapes, image_id, category_ids, width=128,
                                subsampling=1):
    annotations = []

    w = subsampling * width
    b = subsampling
    img = np.zeros((w, w, 3), np.uint8)
    img[:, :] = [215, 215, 215]

    eps = 3
    print(category_ids)
    for si, s in enumerate(shapes):
        annotation = {
            "segmentation": [],
            "area": 0,  # to be filled
            "iscrowd": 0,
            "image_id": image_id,
            "bbox": [],  # to be filled
            "category_id": category_ids[si],
            "id": si
        }

        # rescaling for annotations
        #  [top left x position, top left y position, width, height].
        cx = round(w * s.x)
        cy = round(w * s.y)
        cx_ = cx / b
        cy_ = cy / b
        # print('s.x: ', s.x)
        # print('cx_: ', cx_)

        # not sure if this is the right color for openCV
        rgbcolorvalue = _ImageColor.getrgb(s.color)
        if s.shape == "circle":
            size = 0.5 * 0.6 * math.sqrt(4 * w * s.size * w * s.size / math.pi)
            cx = round(w * s.x)
            cy = round(w * s.y)
            cv2.circle(img, (cx, cy), round(size), rgbcolorvalue, -1)

            bbox = (round(w * s.x - size) - 1, round(w * s.y - size) -
                    1, 2 * size / b + eps, 2 * size / b + eps)
            area = 3.14 * (size / b) * (size / b)

        if s.shape == "triangle":
            r = math.radians(30)
            size = 0.7 * math.sqrt(3) * w * s.size / 3
            dx = size * math.cos(r)
            dy = size * math.sin(r)
            p1 = (round(w * s.x), round(w * s.y - size))
            p2 = (round(w * s.x + dx), round(w * s.y + dy))
            p3 = (round(w * s.x - dx), round(w * s.y + dy))
            points = np.array([p1, p2, p3])
            cv2.fillConvexPoly(img, points, rgbcolorvalue, 1)
            eps_h = size / 4.5
            eps_l = size / 6
            bbox = (round(w * s.x - dx) - 1, round(w * s.y - size) -
                    1, 2 * size / b - eps_l, 2 * size / b - eps_h)
            area = 2 * (dx / 4) * (dy / 4)

        if s.shape == "rectangle":
            size = 0.5 * 0.6 * w * s.size
            xs = round(w * s.x - size)
            ys = round(w * s.y - size)
            xe = round(w * s.x + size)
            ye = round(w * s.y + size)
            cv2.rectangle(img, (xs, ys), (xe, ye), rgbcolorvalue, -1)
            bbox = (xs - 1, ys - 1, 2 * size / b + eps, 2 * size / b + eps)
            area = 4 * (size / b) * (size / b)
        annotation["bbox"] = bbox
        annotation["area"] = area
        annotations.append(annotation)
    return annotations


def kandinskyFigureAsYOLOText(shapes, image_id, category_ids, width=128,
                              subsampling=1):
    annotations = []
    label_texts = []

    w = subsampling * width
    b = subsampling
    img = np.zeros((w, w, 3), np.uint8)
    img[:, :] = [215, 215, 215]

    eps = 3
    print(category_ids)
    for si, s in enumerate(shapes):
        annotation = {
            "segmentation": [],
            "area": 0,  # to be filled
            "iscrowd": 0,
            "image_id": image_id,
            "bbox": [],  # to be filled
            "category_id": category_ids[si],
            "id": si
        }

        # rescaling for annotations
        #  [top left x position, top left y position, width, height].
        cx = round(w * s.x)
        cy = round(w * s.y)
        cx_ = cx / b
        cy_ = cy / b
        # print('s.x: ', s.x)
        # print('cx_: ', cx_)

        # not sure if this is the right color for openCV
        rgbcolorvalue = ImageColor.getrgb(s.color)
        if s.shape == "circle":
            size = 0.5 * 0.6 * math.sqrt(4 * w * s.size * w * s.size / math.pi)
            cx = round(w * s.x)
            cy = round(w * s.y)
            cv2.circle(img, (cx, cy), round(size), rgbcolorvalue, -1)

            bbox = (round(w * s.x) - 1, round(w * s.y) - 1,
                    2 * size / b + eps, 2 * size / b + eps)
            area = 3.14 * (size / b) * (size / b)

        if s.shape == "triangle":
            r = math.radians(30)
            size = 0.7 * math.sqrt(3) * w * s.size / 3
            dx = size * math.cos(r)
            dy = size * math.sin(r)
            p1 = (round(w * s.x), round(w * s.y - size))
            p2 = (round(w * s.x + dx), round(w * s.y + dy))
            p3 = (round(w * s.x - dx), round(w * s.y + dy))
            points = np.array([p1, p2, p3])
            cv2.fillConvexPoly(img, points, rgbcolorvalue, 1)
            eps_h = size / 4.5
            eps_l = size / 6
            bbox = (round(w * s.x) - 1, round(w * s.y) - 1,
                    2 * size / b - eps_l, 2 * size / b - eps_h)
            area = 2 * (dx / 4) * (dy / 4)

        if s.shape == "rectangle":
            size = 0.5 * 0.6 * w * s.size
            xs = round(w * s.x - size)
            ys = round(w * s.y - size)
            xe = round(w * s.x + size)
            ye = round(w * s.y + size)
            cv2.rectangle(img, (xs, ys), (xe, ye), rgbcolorvalue, -1)
            bbox = (w * s.x - 1, w * s.y - 1, 2 * size / b + eps, 2 * size / b + eps)
            area = 4 * (size / b) * (size / b)
        annotation["bbox"] = bbox
        annotation["area"] = area
        annotations.append(annotation)
        label_text = str(category_ids[si]) + " " + str(bbox[0] / w) + " " + str(
            bbox[1] / w) + " " + str(bbox[2] / w) + " " + str(bbox[3] / w)
        print(label_text)
        label_texts.append(label_text)
    return label_texts


def __kandinskyFigureAsYOLOText(shapes, image_id, category_ids, width=128,
                                subsampling=1):
    annotations = []
    label_texts = []
    w = subsampling * width
    b = subsampling
    img = np.zeros((w, w, 3), np.uint8)
    img[:, :] = [150, 150, 150]

    eps = 3
    print(category_ids)
    for si, s in enumerate(shapes):
        annotation = {
            "segmentation": [],
            "area": 0,  # to be filled
            "iscrowd": 0,
            "image_id": image_id,
            "bbox": [],  # to be filled
            "category_id": category_ids[si],
            "id": si
        }

        # rescaling for annotations
        #  [top left x position, top left y position, width, height].
        # print('s.x: ', s.x)
        # print('cx_: ', cx_)

        # not sure if this is the right color for openCV
        rgbcolorvalue = ImageColor.getrgb(s.color)
        if s.shape == "circle":
            size = 0.5 * 0.6 * math.sqrt(4 * s.size * s.size / math.pi)
            cv2.circle(img, (s.x, s.y), round(size), rgbcolorvalue, -1)
            # label_idx x_center y_center width height
            label_text = str(category_id) + " " + str(s.x - size) + " " + \
                         str(y.x - size) + " " + str(2 * size) + " " + str(2 * size)
            label_texts.append(label_text)

        if s.shape == "triangle":
            r = math.radians(30)
            size = 0.7 * math.sqrt(3) * s.size / 3
            dx = size * math.cos(r)
            dy = size * math.sin(r)
            p1 = (s.x, s.y - size)
            p2 = (s.x + dx, s.y + dy)
            p3 = (s.x - dx, s.y + dy)
            points = np.array([p1, p2, p3])
            cv2.fillConvexPoly(img, points, rgbcolorvalue, 1)
            label_text = str(category_id) + " " + str(s.x - dx) + " " + \
                         str(s.y - dy) + " " + str(2 * size) + " " + str(2 * size)
            label_texts.append(label_text)

        if s.shape == "rectangle":
            size = 0.5 * 0.6 * s.size
            xs = s.x - size
            ys = s.y - size
            xe = s.x + size
            ye = s.y + size
            cv2.rectangle(img, (xs, ys), (xe, ye), rgbcolorvalue, -1)
            label_text = str(category_id) + " " + str(xs) + " " + \
                         str(ys) + " " + str(2 * size) + " " + str(2 * size)
            label_texts.append(label_text)

    return label_texts


def overlaps(shapes, width=1024):
    image = Image.new("L", (width, width), 0)
    sumarray = np.array(image)
    d = ImageDraw.Draw(image)
    w = width

    for s in shapes:
        image = Image.new("L", (width, width), 0)
        d = ImageDraw.Draw(image)
        try:
            globals()[s.shape](d, w * s.x, w * s.y, w * s.size, 10)
        except KeyError:
            raise KeyError
        sumarray = sumarray + np.array(image)

    sumimage = Image.fromarray(sumarray)
    return sumimage.getextrema()[1] > 10


def overflow(shapes):
    for s in shapes:
        if not (0.05 < s.x < 0.95 and 0.05 < s.y < 0.95):
            return True
    return False


# matplotlib_colors = {k: tuple(int(v[i:i + 2], 16) for i in (1, 3, 5)) for k, v in
#                      list(matplotlib.colors.cnames.items())}
# matplotlib_colors.pop("black")
# matplotlib_colors_list = [k for k, v in matplotlib_colors.items()]
#
kandinsky_shapes = ['rectangle', 'circle', 'triangle']
