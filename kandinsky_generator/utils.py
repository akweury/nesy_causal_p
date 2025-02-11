# Created by X at 04.11.23
import math


def draw_obj(draw, shape, obj_size, obj_color, obj_x, obj_y):
    if shape == "circle":
        obj_size = 0.7 * obj_size * 4 / math.pi
        draw.ellipse(((obj_x - obj_size / 2, obj_y - obj_size / 2), (obj_x + obj_size / 2, obj_y + obj_size / 2)),
                     fill=obj_color)
    elif shape == "square":
        obj_size = 0.7 * obj_size
        draw.rectangle(((obj_x - obj_size / 2, obj_y - obj_size / 2), (obj_x + obj_size / 2, obj_y + obj_size / 2)),
                       fill=obj_color)
    elif shape == "triangle":
        r = math.radians(30)
        # correct the size to  the same area as an square
        obj_size = 0.7 * obj_size * 3 * math.sqrt(3) / 4
        dx = obj_size * math.cos(r) / 2
        dy = obj_size * math.sin(r) / 2
        draw.polygon([(obj_x, obj_y - obj_size / 2), (obj_x + dx, obj_y + dy), (obj_x - dx, obj_y + dy)],
                     fill=obj_color)
    else:
        raise ValueError
