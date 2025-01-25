import logging
import random
import math
from tqdm import tqdm
import numpy as np

from kandinsky_generator.src.kp.KandinskyTruth import KandinskyTruthInterfce
from kandinsky_generator.src.kp import KandinskyUniverse
from kandinsky_generator.src.kp.RandomKandinskyFigure import Random
from src import bk


class ShapeOnShape(KandinskyTruthInterfce):

    def humanDescription(self):
        return "shapes are on shapes but never on the same"

    def _bigCircle(self, so, t):
        kf = []
        x = 0.4 + random.random() * 0.2
        y = 0.4 + random.random() * 0.2
        r = 0.3 - min(abs(0.5 - x), abs(0.5 - y))
        n = int(2 * r * math.pi / 0.2)

        if n < self.min:   n = self.min
        if n > self.max:   n = self.max

        for i in range(n):
            o = KandinskyUniverse.kandinskyShape()
            d = i * 2 * math.pi / n
            if t:
                o.color = random.choice(["blue", "yellow"])
                o.shape = random.choice(["square", "triangle"])
            else:
                o.color = random.choice(bk.color_large)
                o.shape = random.choice(KandinskyUniverse.kandinsky_shapes)

            o.size = so
            o.x = x + r * math.cos(d)
            o.y = y + r * math.sin(d)
            o.solid = True
            o.line_width = -1
            kf.append(o)
        return kf

    def _circle(self, so, lw, min_percent=1.0, max_percent=1.0):
        kf = []
        x = 0.5  # + random.random() * 0.8
        y = 0.7  # + random.random() * 0.8
        r = 0.3 - min(abs(0.5 - x), abs(0.5 - y)) * 0.5

        xs = x
        ys = y - r

        # so = min_so + random.random() * 0.8

        o = KandinskyUniverse.kandinskyShape()
        o.color = "lightblue"
        o.shape = "circle"
        o.line_width = lw
        o.size = so
        o.x = xs
        o.y = ys
        kf.append(o)

        random_percent = random.uniform(min_percent, max_percent)
        kf = kf[:int(len(kf) * random_percent)]

        return kf

    def _square(self, so, lw, min_percent=1.0, max_percent=1.0):
        kf = []
        x = 0.5  # + random.random() * 0.8
        y = 0.8  # + random.random() * 0.8
        r = 0.3 - min(abs(0.5 - x), abs(0.5 - y)) * 0.5

        xs = x
        ys = y - r

        o = KandinskyUniverse.kandinskyShape()
        o.color = "darkblue"
        o.shape = "square"
        o.line_width = lw
        o.size = so
        o.x = xs
        o.y = ys
        kf.append(o)

        random_percent = random.uniform(min_percent, max_percent)
        kf = kf[:int(len(kf) * random_percent)]
        return kf

    def _bigTriangle(self, so, t, min_percent=1.0, max_percent=1.0):

        kf = []
        x = 0.4 + random.random() * 0.2
        y = 0.4 + random.random() * 0.2
        r = 0.3 - min(abs(0.5 - x), abs(0.5 - y))
        n = int(2 * r * math.pi / 0.25)

        innerdegree = math.radians(30)
        dx = r * math.cos(innerdegree)
        dy = r * math.sin(innerdegree)

        if n < self.min:   n = self.min
        if n > self.max:   n = self.max

        n = round(n / 3)

        xs = x
        ys = y - r
        xe = x + dx
        ye = y + dy
        dxi = (xe - xs) / n
        dyi = (ye - ys) / n

        for i in range(n + 1):
            o = KandinskyUniverse.kandinskyShape()
            if t:
                o.color = random.choice(["yellow", "green"])
                if o.color == "yellow":
                    o.shape = "square"
                else:
                    o.shape = "circle"
            else:
                o.color = random.choice(KandinskyUniverse.matplotlib_colors_list)
                o.shape = random.choice(KandinskyUniverse.kandinsky_shapes)
            o.size = so
            o.solid = True
            o.line_width = -1
            o.x = xs + i * dxi
            o.y = ys + i * dyi
            kf.append(o)

        xs = x + dx
        ys = y + dy
        xe = x - dx
        ye = y + dy
        dxi = (xe - xs) / n
        dyi = (ye - ys) / n

        for i in range(n):
            o = KandinskyUniverse.kandinskyShape()
            if t:
                o.color = random.choice(["yellow", "green"])
                if o.color == "yellow":
                    o.shape = "square"
                else:
                    o.shape = "circle"
            else:
                o.color = random.choice(KandinskyUniverse.matplotlib_colors_list)
                o.shape = random.choice(KandinskyUniverse.kandinsky_shapes)
            o.size = so
            o.solid = True
            o.line_width = -1
            o.x = xs + (i + 1) * dxi
            o.y = ys + (i + 1) * dyi
            kf.append(o)

        xs = x - dx
        ys = y + dy
        xe = x
        ye = y - r
        dxi = (xe - xs) / n
        dyi = (ye - ys) / n

        for i in range(n - 1):
            o = KandinskyUniverse.kandinskyShape()
            if t:
                o.color = random.choice(["yellow", "green"])
                if o.color == "yellow":
                    o.shape = "square"
                else:
                    o.shape = "circle"
            else:
                o.color = random.choice(KandinskyUniverse.matplotlib_colors_list)
                o.shape = random.choice(KandinskyUniverse.kandinsky_shapes)
            o.size = so
            o.solid = True
            o.line_width = -1
            o.x = xs + (i + 1) * dxi
            o.y = ys + (i + 1) * dyi
            kf.append(o)

        random_percent = random.uniform(min_percent, max_percent)
        kf = kf[:int(len(kf) * random_percent)]

        return kf

    def _bigTriangleCF(self, so, t, min_percent=1.0, max_percent=1.0):

        kf = []
        x = 0.4 + random.random() * 0.2
        y = 0.4 + random.random() * 0.2
        r = 0.3 - min(abs(0.5 - x), abs(0.5 - y))
        n = int(2 * r * math.pi / 0.25)

        innerdegree = math.radians(30)
        dx = r * math.cos(innerdegree)
        dy = r * math.sin(innerdegree)

        if n < self.min:   n = self.min
        if n > self.max:   n = self.max

        n = round(n / 3)

        xs = x
        ys = y - r
        xe = x + dx
        ye = y + dy
        dxi = (xe - xs) / n
        dyi = (ye - ys) / n

        for i in range(n + 1):
            o = KandinskyUniverse.kandinskyShape()

            o.color = random.choice(["yellow", "green"])
            if o.color == "yellow":
                o.shape = "circle"
            else:
                o.shape = "square"

            o.size = so
            o.x = xs + i * dxi
            o.y = ys + i * dyi
            o.solid = True
            o.line_width = -1
            kf.append(o)

        xs = x + dx
        ys = y + dy
        xe = x - dx
        ye = y + dy
        dxi = (xe - xs) / n
        dyi = (ye - ys) / n

        for i in range(n):
            o = KandinskyUniverse.kandinskyShape()

            o.color = random.choice(["yellow", "green"])
            if o.color == "yellow":
                o.shape = "circle"
            else:
                o.shape = "square"

            o.size = so
            o.x = xs + (i + 1) * dxi
            o.y = ys + (i + 1) * dyi
            o.solid = True
            o.line_width = -1
            kf.append(o)

        xs = x - dx
        ys = y + dy
        xe = x
        ye = y - r
        dxi = (xe - xs) / n
        dyi = (ye - ys) / n

        for i in range(n - 1):
            o = KandinskyUniverse.kandinskyShape()

            o.color = random.choice(["yellow", "green"])
            if o.color == "yellow":
                o.shape = "circle"
            else:
                o.shape = "square"

            o.size = so
            o.x = xs + (i + 1) * dxi
            o.y = ys + (i + 1) * dyi
            o.solid = True
            o.line_width = -1
            kf.append(o)

        random_percent = random.uniform(min_percent, max_percent)
        kf = kf[:int(len(kf) * random_percent)]

        return kf

    def _bigTriangleCF2(self, so, t, min_percent=1.0, max_percent=1.0):

        kf = []
        x = 0.4 + random.random() * 0.2
        y = 0.4 + random.random() * 0.2
        r = 0.3 - min(abs(0.5 - x), abs(0.5 - y))
        n = int(2 * r * math.pi / 0.25)

        innerdegree = math.radians(30)
        dx = r * math.cos(innerdegree)
        dy = r * math.sin(innerdegree)

        if n < self.min:   n = self.min
        if n > self.max:   n = self.max

        n = round(n / 3)

        xs = x
        ys = y - r
        xe = x + dx
        ye = y + dy
        dxi = (xe - xs) / n
        dyi = (ye - ys) / n

        for i in range(n + 1):
            o = KandinskyUniverse.kandinskyShape()
            o.color = random.choice(["yellow", "green"])
            if o.color == "yellow":
                o.shape = "square"
            else:
                o.shape = "circle"

            o.size = so
            o.x = xs + i * dxi
            o.y = ys + i * dyi
            o.solid = True
            o.line_width = -1
            kf.append(o)

        xs = x + dx
        ys = y + dy
        xe = x - dx
        ye = y + dy
        dxi = (xe - xs) / n
        dyi = (ye - ys) / n

        for i in range(n):
            o = KandinskyUniverse.kandinskyShape()

            o.color = random.choice(["yellow", "green"])
            if o.color == "yellow":
                o.shape = "square"
            else:
                o.shape = "circle"

            o.size = so
            o.x = xs + (i + 1) * dxi
            o.y = ys + (i + 1) * dyi
            o.solid = True
            o.line_width = -1
            kf.append(o)

        xs = x - dx
        ys = y + dy
        xe = x
        ye = y - r
        dxi = (xe - xs) / n
        dyi = (ye - ys) / n

        # for i in range(n - 1):
        #     o = KandinskyUniverse.kandinskyShape()
        #     if t:
        #         o.color = random.choice(["yellow", "green"])
        #         if o.color=="yellow":
        #             o.shape="square"
        #         else:
        #             o.shape="circle"
        #     else:
        #         o.color = random.choice(KandinskyUniverse.matplotlib_colors_list)
        #         o.shape = random.choice(KandinskyUniverse.kandinsky_shapes)
        #     o.size = so
        #     o.x = xs + (i + 1) * dxi
        #     o.y = ys + (i + 1) * dyi
        #     kf.append(o)

        random_percent = random.uniform(min_percent, max_percent)
        kf = kf[:int(len(kf) * random_percent)]

        return kf

    def _triangle(self, so, lw, min_percent=1.0, max_percent=1.0):
        kf = []
        x = 0.2  # + random.random() * 0.8
        y = 0.8  # + random.random() * 0.8

        o = KandinskyUniverse.kandinskyShape()
        o.color = "green"
        o.shape = "triangle"
        o.size = so
        o.line_width = lw
        o.x = x
        o.y = y
        kf.append(o)
        return kf

    def _smallCircleFlex(self, so, t):
        kf = []
        # x, y in range [0.1, 0.1 + 0.8]
        x = 0.1 + random.random() * 0.8
        y = 0.1 + random.random() * 0.8
        # r in range [0]
        r = 0.2 - min(abs(0.5 - x), abs(0.5 - y)) * 0.5
        n = int(10 * r * math.pi / 0.2)

        if n < 5:   n = 5
        if n > 15:   n = 15

        random_rotate_rad = random.random()
        for i in range(n):
            o = KandinskyUniverse.kandinskyShape()
            d = (i + random_rotate_rad) * 2 * math.pi / n
            if t:
                o.color = random.choice(["blue", "yellow"])
                o.shape = random.choice(["square", "triangle"])
            else:
                o.color = random.choice(KandinskyUniverse.matplotlib_colors_list)
                o.shape = random.choice(KandinskyUniverse.kandinsky_shapes)

            o.size = so
            o.x = x + r * math.cos(d)
            o.y = y + r * math.sin(d)
            kf.append(o)
        return kf

    def _bigCircleFlex(self, so, t):
        kf = []
        x = 0.4 + random.random() * 0.2
        y = 0.4 + random.random() * 0.2
        r = 0.3 - min(abs(0.5 - x), abs(0.5 - y))
        n = int(2 * r * math.pi / 0.2)

        if n < self.min:   n = self.min
        if n > self.max:   n = self.max

        random_rotate_rad = random.random()
        for i in range(n):
            o = KandinskyUniverse.kandinskyShape()
            d = (i + random_rotate_rad) * 2 * math.pi / n
            if t:
                o.color = random.choice(["blue", "yellow"])
                o.shape = random.choice(["square", "triangle"])
            else:
                o.color = random.choice(KandinskyUniverse.matplotlib_colors_list)
                o.shape = random.choice(KandinskyUniverse.kandinsky_shapes)

            o.size = so
            o.x = x + r * math.cos(d)
            o.y = y + r * math.sin(d)
            kf.append(o)
        return kf

    def _triangleSolidBig(self, so, t, min_percent=1.0, max_percent=1.0):
        kf = []
        x = 0.5  # + random.random() * 0.8
        y = 0.7  # + random.random() * 0.8
        r = 0.3 - min(abs(0.5 - x), abs(0.5 - y)) * 0.5

        xs = x
        ys = y - r

        so = 0.5 + random.random() * 0.3

        o = KandinskyUniverse.kandinskyShape()
        o.color = random.choice(KandinskyUniverse.matplotlib_colors_list)
        o.shape = "triangle"
        o.size = so
        o.x = xs
        o.y = ys
        kf.append(o)

        random_percent = random.uniform(min_percent, max_percent)
        kf = kf[:int(len(kf) * random_percent)]
        return kf

    def _gestaltTriangle(self, so, t, min_percent=1.0, max_percent=1.0):
        kf = []
        x = 0.5  # + random.random() * 0.8
        y = 0.8  # + random.random() * 0.8
        r = 0.3 - min(abs(0.5 - x), abs(0.5 - y)) * 0.5
        xs = x
        ys = y - r

        so = 0.4 + random.random() * 0.6
        cir_so = so * 0.3

        # correct the size to  the same area as an square

        s = 0.7 * math.sqrt(3) * so / 3
        dx = s * math.cos(math.radians(30))
        dy = s * math.sin(math.radians(30))
        # draw circles
        o = KandinskyUniverse.kandinskyShape()
        o.color = random.choice(["blue", "green", "yellow"])
        o.shape = "circle"
        o.size = cir_so
        o.solid = True
        o.line_width = -1
        # (cx - s / 2, cy - s / 2), (cx + s / 2, cy + s / 2)
        o.x = xs
        o.y = ys - s
        kf.append(o)

        o = KandinskyUniverse.kandinskyShape()
        o.color = random.choice(["blue", "green", "yellow"])
        o.shape = "circle"
        o.size = cir_so
        o.x = xs + dx
        o.y = ys + dy
        o.solid = True
        o.line_width = -1
        kf.append(o)

        o = KandinskyUniverse.kandinskyShape()
        o.color = random.choice(["blue", "green", "yellow"])
        o.shape = "circle"
        o.size = cir_so
        o.x = xs - dx
        o.y = ys + dy
        o.solid = True
        o.line_width = -1
        kf.append(o)

        # draw triangle
        o = KandinskyUniverse.kandinskyShape()
        o.color = "lightgray"
        o.shape = "triangle"
        o.size = so
        o.x = xs
        o.y = ys
        kf.append(o)

        random_percent = random.uniform(min_percent, max_percent)
        kf = kf[:int(len(kf) * random_percent)]
        return kf

    def _gestaltTriangleCF(self, so, t, min_percent=1.0, max_percent=1.0):
        kf = []
        x = 0.5  # + random.random() * 0.8
        y = 0.8  # + random.random() * 0.8
        r = 0.3 - min(abs(0.5 - x), abs(0.5 - y)) * 0.5
        xs = x
        ys = y - r

        so = 0.4 + random.random() * 0.6
        cir_so = so * 0.3

        # correct the size to  the same area as an square

        s = 0.7 * math.sqrt(3) * so / 3
        dx = s * math.cos(math.radians(30))
        dy = s * math.sin(math.radians(30))
        # draw circles
        o = KandinskyUniverse.kandinskyShape()
        o.color = random.choice(["blue", "green", "yellow"])
        o.shape = "circle"
        o.size = cir_so
        o.solid = True
        o.line_width = -1
        # (cx - s / 2, cy - s / 2), (cx + s / 2, cy + s / 2)
        o.x = xs
        o.y = ys - s
        kf.append(o)

        o = KandinskyUniverse.kandinskyShape()
        o.color = random.choice(["blue", "green", "yellow"])
        o.shape = "circle"
        o.size = cir_so
        o.x = xs + dx
        o.y = ys + dy
        o.solid = True
        o.line_width = -1
        kf.append(o)

        o = KandinskyUniverse.kandinskyShape()
        o.color = random.choice(["blue", "green", "yellow"])
        o.shape = "circle"
        o.size = cir_so
        o.x = xs - dx
        o.y = ys + dy
        o.solid = True
        o.line_width = -1
        kf.append(o)

        # # draw triangle
        # o = KandinskyUniverse.kandinskyShape()
        # o.color = "lightgray"
        # o.shape = "triangle"
        # o.size = so
        # o.x = xs
        # o.y = ys
        # kf.append(o)

        random_percent = random.uniform(min_percent, max_percent)
        kf = kf[:int(len(kf) * random_percent)]
        return kf

    def _gestaltCircleTriangle(self, so, t, min_percent=1.0, max_percent=1.0):
        so = so * 3
        # draw big circle
        kf = []
        x = 0.4 + random.random() * 0.2
        y = 0.4 + random.random() * 0.2
        r = 0.3 - min(abs(0.5 - x), abs(0.5 - y))
        n = 7

        # if n < self.min:   n = self.min
        # if n > self.max:   n = self.max

        for i in range(n):
            o = KandinskyUniverse.kandinskyShape()
            d = i * 2 * math.pi / n
            if t:
                o.color = random.choice(["blue", "yellow"])
                o.shape = random.choice(["square", "triangle"])
            else:
                o.color = random.choice(bk.color_large)
                o.shape = random.choice(KandinskyUniverse.kandinsky_shapes)

            o.size = so
            o.x = x + r * math.cos(d)
            o.y = y + r * math.sin(d)
            o.solid = True
            o.line_width = -1
            kf.append(o)

        # draw triangle
        o = KandinskyUniverse.kandinskyShape()
        o.color = "green"
        o.shape = "triangle"
        o.size = 2 * r * 1.4
        o.x = x
        o.y = y
        kf.append(o)

        random_percent = random.uniform(min_percent, max_percent)
        kf = kf[:int(len(kf) * random_percent)]
        return kf

    def _proximity_square(self, so, t):
        objs = []
        so = 0.1

        mode = random.choice([1, 2])
        if mode == 1:
            for x in range(1, 10, 2):
                for y in range(1, 10, 2):
                    if (x in [1, 3] and y in [1, 3]) or (x in [7, 9] and y in [7, 9]):
                        # draw triangle
                        # draw triangle
                        objs.append(KandinskyUniverse.kandinskyShape(
                            color=random.choice(bk.color_large_exclude_gray),
                            shape=random.choice(KandinskyUniverse.kandinsky_shapes),
                            size=so,
                            x=x * 0.1,
                            y=y * 0.1,
                            line_width=-1,
                            solid=True))
        else:
            for x in range(1, 10, 2):
                for y in range(1, 10, 2):
                    if (x in [1, 3] and y in [7, 9]) or (x in [7, 9] and y in [1, 3]):
                        # draw triangle
                        # draw triangle
                        objs.append(KandinskyUniverse.kandinskyShape(
                            color=random.choice(bk.color_large_exclude_gray),
                            shape=random.choice(KandinskyUniverse.kandinsky_shapes),
                            size=so,
                            x=x * 0.1,
                            y=y * 0.1,
                            line_width=-1,
                            solid=True))
        return objs

    def _proximity_squareCF(self, so, t):
        objs = []
        so = 0.1

        mode = random.choice([1, 2])
        if mode == 1:
            for x in range(1, 10, 2):
                for y in range(1, 10, 2):
                    if (x in [1, 3] and y in [1, 3]) or (x in [5, 7] and y in [1, 3]):
                        # draw triangle
                        # draw triangle
                        objs.append(KandinskyUniverse.kandinskyShape(
                            color=random.choice(bk.color_large_exclude_gray),
                            shape=random.choice(KandinskyUniverse.kandinsky_shapes),
                            size=so,
                            x=x * 0.1,
                            y=y * 0.1,
                            line_width=-1,
                            solid=True))
        else:
            for x in range(1, 10, 2):
                for y in range(1, 10, 2):
                    if (x in [1, 3] and y in [5, 7]) or (x in [5, 7] and y in [5, 7]):
                        # draw triangle
                        # draw triangle
                        objs.append(KandinskyUniverse.kandinskyShape(
                            color=random.choice(bk.color_large_exclude_gray),
                            shape=random.choice(KandinskyUniverse.kandinsky_shapes),
                            size=so,
                            x=x * 0.1,
                            y=y * 0.1,
                            line_width=-1,
                            solid=True))
        # x = 0.5  # + random.random() * 0.8
        # y = 0.7  # + random.random() * 0.8
        # r = 0.3 - min(abs(0.5 - x), abs(0.5 - y)) * 0.5
        # xs = x
        # ys = y - r
        #
        # so = 0.4 + random.random() * 0.6
        # cir_so = so * 0.3
        #
        # # correct the size to  the same area as an square
        # s = 0.7 * math.sqrt(3) * so / 3
        # dx = s * math.cos(math.radians(30))
        # dy = s * math.sin(math.radians(30))
        #
        # # draw circles
        # o = KandinskyUniverse.kandinskyShape()
        # o.color = random.choice(["blue", "green", "yellow"])
        # o.shape = "circle"
        # o.size = cir_so
        # # (cx - s / 2, cy - s / 2), (cx + s / 2, cy + s / 2)
        # o.x = xs
        # o.y = ys - s
        # objs.append(o)
        #
        # o = KandinskyUniverse.kandinskyShape()
        # o.color = random.choice(["blue", "green", "yellow"])
        # o.shape = "circle"
        # o.size = cir_so
        # o.x = xs + dx
        # o.y = ys + dy
        # objs.append(o)
        #
        # o = KandinskyUniverse.kandinskyShape()
        # o.color = random.choice(["blue", "green", "yellow"])
        # o.shape = "circle"
        # o.size = cir_so
        # o.x = xs - dx
        # o.y = ys + dy
        # objs.append(o)

        return objs

    def _continue_two_curves(self):

        color = random.choice(bk.color_large)
        while color == "lightgray":
            color = random.choice(bk.color_large)

        objs = []
        so = 0.1
        row_num = random.randint(3, 5)
        col_num = random.randint(3, 5)
        diff_row_id = random.randint(0, row_num - 1)
        diff_col_id = random.randint(0, col_num - 1)
        row_space = 1 / (row_num + 1)
        col_space = 1 / (col_num + 1)
        for x in range(row_num):
            for y in range(col_num):
                if x != diff_row_id and y != diff_col_id:
                    # draw triangle
                    objs.append(KandinskyUniverse.kandinskyShape(
                        color=color,
                        shape="triangle",
                        size=so,
                        x=(x + 1) * row_space,
                        y=(y + 1) * col_space,
                        line_width=-1,
                        solid=True))
                else:
                    # draw circle
                    objs.append(KandinskyUniverse.kandinskyShape(
                        color=color,
                        shape="circle",
                        size=so,
                        x=(x + 1) * row_space,
                        y=(y + 1) * col_space,
                        line_width=-1,
                        solid=True))

        return objs

    def _similarity_triangle_circle(self):

        color = random.choice(bk.color_large)
        while color == "lightgray":
            color = random.choice(bk.color_large)

        objs = []
        so = 0.1
        row_num = random.randint(3, 5)
        col_num = random.randint(3, 5)
        diff_row_id = random.randint(0, row_num - 1)
        diff_col_id = random.randint(0, col_num - 1)
        row_space = 1 / (row_num + 1)
        col_space = 1 / (col_num + 1)
        for x in range(row_num):
            for y in range(col_num):
                if x != diff_row_id and y != diff_col_id:
                    # draw triangle
                    objs.append(KandinskyUniverse.kandinskyShape(
                        color=color,
                        shape="triangle",
                        size=so,
                        x=(x + 1) * row_space,
                        y=(y + 1) * col_space,
                        line_width=-1,
                        solid=True))
                else:
                    # draw circle
                    objs.append(KandinskyUniverse.kandinskyShape(
                        color=color,
                        shape="circle",
                        size=so,
                        x=(x + 1) * row_space,
                        y=(y + 1) * col_space,
                        line_width=-1,
                        solid=True))

        return objs

    def _similarity_triangle_circleCF(self):

        color = random.choice(bk.color_large)
        while color == "lightgray":
            color = random.choice(bk.color_large)

        objs = []
        so = 0.1
        row_num = random.randint(3, 5)
        col_num = random.randint(3, 5)
        diff_row_id = random.randint(0, row_num - 1)
        diff_col_id = random.randint(0, col_num - 1)
        row_space = 1 / (row_num + 1)
        col_space = 1 / (col_num + 1)
        for x in range(row_num):
            for y in range(col_num):
                if x != diff_row_id and y != diff_col_id:
                    # draw triangle
                    objs.append(KandinskyUniverse.kandinskyShape(
                        color=color,
                        shape="circle",
                        size=so,
                        x=(x + 1) * row_space,
                        y=(y + 1) * col_space,
                        line_width=-1,
                        solid=True))
                else:
                    # draw circle
                    objs.append(KandinskyUniverse.kandinskyShape(
                        color=color,
                        shape="triangle",
                        size=so,
                        x=(x + 1) * row_space,
                        y=(y + 1) * col_space,
                        line_width=-1,
                        solid=True))

        return objs

    def _smallTriangle(self, so, t, min_percent=1.0, max_percent=1.0):
        kf = []
        x = 0.1 + random.random() * 0.8
        y = 0.1 + random.random() * 0.8
        r = 0.3 - min(abs(0.5 - x), abs(0.5 - y)) * 0.5
        n = int(2 * r * math.pi / 0.25)

        innerdegree = math.radians(30)
        dx = r * math.cos(innerdegree)
        dy = r * math.sin(innerdegree)

        if n < self.min:   n = self.min
        if n > self.max:   n = self.max

        n = round(n / 3)

        xs = x
        ys = y - r
        xe = x + dx
        ye = y + dy

        edge_n = n - random.randint(0, 2)
        dxi = (xe - xs) / edge_n
        dyi = (ye - ys) / edge_n

        first_edge_shift = random.random() * 0.8
        for i in range(edge_n + 1):
            o = KandinskyUniverse.kandinskyShape()
            if t:
                o.color = random.choice(["yellow", "red"])
                o.shape = random.choice(["circle", "square"])
            else:
                o.color = random.choice(KandinskyUniverse.matplotlib_colors_list)
                o.shape = random.choice(KandinskyUniverse.kandinsky_shapes)
            o.size = so
            o.x = xs + i * dxi + first_edge_shift * dxi
            o.y = ys + i * dyi + first_edge_shift * dyi

            if o.x <= xs + edge_n * dxi and o.y <= ys + edge_n * dyi:
                kf.append(o)

        xs = x + dx
        ys = y + dy
        xe = x - dx
        ye = y + dy
        edge_n = n - random.randint(0, 2)
        dxi = (xe - xs) / edge_n
        dyi = (ye - ys) / edge_n

        second_edge_shift = random.random() * 0.8

        for i in range(edge_n):
            o = KandinskyUniverse.kandinskyShape()
            if t:
                o.color = random.choice(["yellow", "red"])
                o.shape = random.choice(["circle", "square"])
            else:
                o.color = random.choice(KandinskyUniverse.matplotlib_colors_list)
                o.shape = random.choice(KandinskyUniverse.kandinsky_shapes)
            o.size = so
            o.x = xs + (i + 1) * dxi - second_edge_shift * dxi
            o.y = ys + (i + 1) * dyi + second_edge_shift * dyi

            if o.x >= xs + (edge_n + 1) * dxi and o.y >= ys + (edge_n + 1) * dyi:
                kf.append(o)

        xs = x - dx
        ys = y + dy
        xe = x
        ye = y - r
        edge_n = n - random.randint(0, 2)
        dxi = (xe - xs) / edge_n
        dyi = (ye - ys) / edge_n

        third_edge_shift = random.random() * 0.8
        for i in range(edge_n - 1):
            o = KandinskyUniverse.kandinskyShape()
            if t:
                o.color = random.choice(["yellow", "red"])
                o.shape = random.choice(["circle", "square"])
            else:
                o.color = random.choice(KandinskyUniverse.matplotlib_colors_list)
                o.shape = random.choice(KandinskyUniverse.kandinsky_shapes)
            o.size = so
            o.x = xs + (i + 1) * dxi - third_edge_shift * dxi
            o.y = ys + (i + 1) * dyi - third_edge_shift * dyi

            if o.x <= xs + (edge_n + 1) * dxi and o.y >= ys + (edge_n + 1) * dyi:
                kf.append(o)

        random_percent = random.uniform(min_percent, max_percent)
        kf = kf[:int(len(kf) * random_percent)]
        return kf

    def _bigTriangler(self, so, t, min_percent=1.0, max_percent=1.0):
        kf = []
        x = 0.4 + random.random() * 0.2
        y = 0.4 + random.random() * 0.2
        r = 0.3 - min(abs(0.5 - x), abs(0.5 - y))
        n = int(2 * r * math.pi / 0.25)

        innerdegree = math.radians(30)
        dx = r * math.cos(innerdegree)
        dy = r * math.sin(innerdegree)

        if n < self.min:   n = self.min
        if n > self.max:   n = self.max

        n = round(n / 3)

        xs = x
        ys = y - r
        xe = x + dx
        ye = y + dy

        edge_n = n - random.randint(0, 2)
        dxi = (xe - xs) / edge_n
        dyi = (ye - ys) / edge_n

        first_edge_shift = random.random() * 0.8
        for i in range(edge_n + 1):
            o = KandinskyUniverse.kandinskyShape()
            if t:
                o.color = random.choice(["yellow", "red"])
                o.shape = random.choice(["circle", "square"])
            else:
                o.color = random.choice(KandinskyUniverse.matplotlib_colors_list)
                o.shape = random.choice(KandinskyUniverse.kandinsky_shapes)
            o.size = so
            o.x = xs + i * dxi + first_edge_shift * dxi
            o.y = ys + i * dyi + first_edge_shift * dyi

            if o.x <= xs + edge_n * dxi and o.y <= ys + edge_n * dyi:
                kf.append(o)

        xs = x + dx
        ys = y + dy
        xe = x - dx
        ye = y + dy
        edge_n = n - random.randint(0, 2)
        dxi = (xe - xs) / edge_n
        dyi = (ye - ys) / edge_n

        second_edge_shift = random.random() * 0.8

        for i in range(edge_n):
            o = KandinskyUniverse.kandinskyShape()
            if t:
                o.color = random.choice(["yellow", "red"])
                o.shape = random.choice(["circle", "square"])
            else:
                o.color = random.choice(KandinskyUniverse.matplotlib_colors_list)
                o.shape = random.choice(KandinskyUniverse.kandinsky_shapes)
            o.size = so
            o.x = xs + (i + 1) * dxi - second_edge_shift * dxi
            o.y = ys + (i + 1) * dyi + second_edge_shift * dyi

            if o.x >= xs + (edge_n + 1) * dxi and o.y >= ys + (edge_n + 1) * dyi:
                kf.append(o)

        xs = x - dx
        ys = y + dy
        xe = x
        ye = y - r
        edge_n = n - random.randint(0, 2)
        dxi = (xe - xs) / edge_n
        dyi = (ye - ys) / edge_n

        third_edge_shift = random.random() * 0.8
        for i in range(edge_n - 1):
            o = KandinskyUniverse.kandinskyShape()
            if t:
                o.color = random.choice(["yellow", "red"])
                o.shape = random.choice(["circle", "square"])
            else:
                o.color = random.choice(KandinskyUniverse.matplotlib_colors_list)
                o.shape = random.choice(KandinskyUniverse.kandinsky_shapes)
            o.size = so
            o.x = xs + (i + 1) * dxi - third_edge_shift * dxi
            o.y = ys + (i + 1) * dyi - third_edge_shift * dyi

            if o.x <= xs + (edge_n + 1) * dxi and o.y >= ys + (edge_n + 1) * dyi:
                kf.append(o)

        random_percent = random.uniform(min_percent, max_percent)
        kf = kf[:int(len(kf) * random_percent)]
        return kf

    def _smallSquare(self, so, t, min_percent=1.0, max_percent=1.0):
        kf = []
        x = 0.1 + random.random() * 0.8
        y = 0.1 + random.random() * 0.8
        r = 0.3 - min(abs(0.5 - x), abs(0.5 - y)) * 0.5
        m = 4 * round(r / 0.05)
        if m < 8:   m = 8
        if m > 20:   m = 20

        minx = x - r / 2
        maxx = x + r / 2
        miny = y - r / 2
        maxy = y + r / 2

        n = int(m / 4)

        dx = r / n
        for i in range(n + 1):
            o = KandinskyUniverse.kandinskyShape()
            if t:
                o.color = random.choice(["blue", "red"])
                o.shape = random.choice(["circle", "triangle"])
            else:
                o.color = random.choice(KandinskyUniverse.matplotlib_colors_list)
                o.shape = random.choice(KandinskyUniverse.kandinsky_shapes)
            o.size = so
            o.x = minx + i * dx
            o.y = miny
            kf.append(o)
            o = KandinskyUniverse.kandinskyShape()
            if t:
                o.color = random.choice(["blue", "red"])
                o.shape = random.choice(["circle", "triangle"])
            else:
                o.color = random.choice(KandinskyUniverse.matplotlib_colors_list)
                o.shape = random.choice(KandinskyUniverse.kandinsky_shapes)
            o.size = so
            o.x = minx + i * dx
            o.y = maxy
            kf.append(o)

        for i in range(n - 1):
            o = KandinskyUniverse.kandinskyShape()
            if t:
                o.color = random.choice(["blue", "red"])
                o.shape = random.choice(["circle", "triangle"])
            else:
                o.color = random.choice(KandinskyUniverse.matplotlib_colors_list)
                o.shape = random.choice(KandinskyUniverse.kandinsky_shapes)
            o.size = so
            o.x = minx
            o.y = miny + (i + 1) * dx
            kf.append(o)
            o = KandinskyUniverse.kandinskyShape()
            if t:
                o.color = random.choice(["blue", "red"])
                o.shape = random.choice(["circle", "triangle"])
            else:
                o.color = random.choice(KandinskyUniverse.matplotlib_colors_list)
                o.shape = random.choice(KandinskyUniverse.kandinsky_shapes)
            o.size = so
            o.x = maxx
            o.y = miny + (i + 1) * dx
            kf.append(o)
        random_percent = random.uniform(min_percent, max_percent)
        kf = kf[:int(len(kf) * random_percent)]
        return kf

    def _bigSquare(self, so, t, min_percent=1.0, max_percent=1.0):
        kf = []
        x = 0.4 + random.random() * 0.2
        y = 0.4 + random.random() * 0.2
        r = 0.4 - min(abs(0.5 - x), abs(0.5 - y))
        m = 4 * int(r / 0.1)
        if m < self.min:   m = self.min
        if m > self.max:   m = self.max
        minx = x - r / 2
        maxx = x + r / 2
        miny = y - r / 2
        maxy = y + r / 2
        n = int(m / 4)
        dx = r / n
        for i in range(n + 1):
            o = KandinskyUniverse.kandinskyShape()
            if t:
                o.color = random.choice(["blue", "red"])
                if o.color == "blue":
                    o.shape = "circle"
                else:
                    o.shape = "square"
            else:
                o.color = random.choice(bk.color_large)
                o.shape = random.choice(KandinskyUniverse.kandinsky_shapes)
            o.size = so
            o.x = minx + i * dx
            o.y = miny
            o.solid = True
            kf.append(o)
            o = KandinskyUniverse.kandinskyShape()
            if t:
                o.color = random.choice(["blue", "red"])
                if o.color == "blue":
                    o.shape = "circle"
                else:
                    o.shape = "square"
            else:
                o.color = random.choice(bk.color_large)
                o.shape = random.choice(KandinskyUniverse.kandinsky_shapes)
            o.size = so
            o.x = minx + i * dx
            o.y = maxy
            o.solid = True
            kf.append(o)

        for i in range(n - 1):
            o = KandinskyUniverse.kandinskyShape()
            if t:
                o.color = random.choice(["blue", "red"])
                if o.color == "blue":
                    o.shape = "circle"
                else:
                    o.shape = "square"
            else:
                o.color = random.choice(bk.color_large)
                o.shape = random.choice(KandinskyUniverse.kandinsky_shapes)
            o.size = so
            o.x = minx
            o.y = miny + (i + 1) * dx
            o.solid = True
            kf.append(o)
            o = KandinskyUniverse.kandinskyShape()
            if t:
                o.color = random.choice(["blue", "red"])
                if o.color == "blue":
                    o.shape = "circle"
                else:
                    o.shape = "square"
            else:
                o.color = random.choice(bk.color_large)
                o.shape = random.choice(KandinskyUniverse.kandinsky_shapes)
            o.size = so
            o.x = maxx
            o.y = miny + (i + 1) * dx
            o.solid = True
            kf.append(o)
        random_percent = random.uniform(min_percent, max_percent)
        kf = kf[:int(len(kf) * random_percent)]
        return kf

    def _bigSquareCF(self, so, t, min_percent=1.0, max_percent=1.0):
        kf = []
        x = 0.4 + random.random() * 0.2
        y = 0.4 + random.random() * 0.2
        r = 0.4 - min(abs(0.5 - x), abs(0.5 - y))
        m = 4 * int(r / 0.1)
        if m < self.min:   m = self.min
        if m > self.max:   m = self.max
        minx = x - r / 2
        maxx = x + r / 2
        miny = y - r / 2
        maxy = y + r / 2
        n = int(m / 4)
        dx = r / n
        for i in range(n + 1):
            o = KandinskyUniverse.kandinskyShape()
            o.color = random.choice(["blue", "red"])
            if o.color == "blue":
                o.shape = "square"
            else:
                o.shape = "circle"
            o.size = so
            o.x = minx + i * dx
            o.y = miny
            o.solid = True
            kf.append(o)
            o = KandinskyUniverse.kandinskyShape()
            o.color = random.choice(["blue", "red"])
            if o.color == "blue":
                o.shape = "square"
            else:
                o.shape = "circle"
            o.size = so
            o.x = minx + i * dx
            o.y = maxy
            o.solid = True
            kf.append(o)

        for i in range(n - 1):
            o = KandinskyUniverse.kandinskyShape()
            o.color = random.choice(["blue", "red"])
            if o.color == "blue":
                o.shape = "square"
            else:
                o.shape = "circle"
            o.size = so
            o.x = minx
            o.y = miny + (i + 1) * dx
            o.solid = True
            kf.append(o)
            o = KandinskyUniverse.kandinskyShape()
            o.color = random.choice(["blue", "red"])
            if o.color == "blue":
                o.shape = "square"
            else:
                o.shape = "circle"
            o.size = so
            o.x = maxx
            o.y = miny + (i + 1) * dx
            o.solid = True
            kf.append(o)
        random_percent = random.uniform(min_percent, max_percent)
        kf = kf[:int(len(kf) * random_percent)]
        return kf

    def _bigSquareCF2(self, so, t, min_percent=1.0, max_percent=1.0):
        kf = []
        x = 0.4 + random.random() * 0.2
        y = 0.4 + random.random() * 0.2
        r = 0.4 - min(abs(0.5 - x), abs(0.5 - y))
        m = 4 * int(r / 0.1)
        if m < self.min:   m = self.min
        if m > self.max:   m = self.max
        minx = x - r / 2
        maxx = x + r / 2
        miny = y - r / 2
        maxy = y + r / 2
        n = int(m / 4)
        dx = r / n
        for i in range(n + 1):
            o = KandinskyUniverse.kandinskyShape()

            o.color = random.choice(["blue", "red"])
            if o.color == "blue":
                o.shape = "circle"
            else:
                o.shape = "square"
            o.size = so
            o.x = minx + i * dx
            o.y = miny
            o.solid = True
            kf.append(o)

        for i in range(n - 1):
            o = KandinskyUniverse.kandinskyShape()

            o.color = random.choice(["blue", "red"])
            if o.color == "blue":
                o.shape = "circle"
            else:
                o.shape = "square"

            o.size = so
            o.x = minx
            o.y = miny + (i + 1) * dx
            o.solid = True
            kf.append(o)
            o = KandinskyUniverse.kandinskyShape()

            o.color = random.choice(["blue", "red"])
            if o.color == "blue":
                o.shape = "circle"
            else:
                o.shape = "square"

            o.size = so
            o.x = maxx
            o.y = miny + (i + 1) * dx
            o.solid = True
            kf.append(o)
        random_percent = random.uniform(min_percent, max_percent)
        kf = kf[:int(len(kf) * random_percent)]
        return kf

    def _bigDiamond(self, so, t, min_percent=1.0, max_percent=1.0):
        """
        Generates n points that form a diamond shape.

        :param center: Tuple (x, y) representing the center of the diamond.
        :param scale: Scale factor to control the size of the diamond.
        :param n: Number of points to generate along the diamond's edges.
        :return: A NumPy array of shape (n, 2) with the (x, y) coordinates of the diamond points.
        """
        n = 20

        x = 0.4 + random.random() * 0.2
        y = 0.4 + random.random() * 0.2
        scale = 0.4 - min(abs(0.5 - x), abs(0.5 - y))

        # Define the 4 key corners of the diamond
        top = (x, y - scale)  # Top point
        right = (x + scale, y)  # Right point
        bottom = (x, y + scale)  # Bottom point
        left = (x - scale, y)  # Left point

        # Divide n points into 4 segments (1 for each edge)
        points_per_edge = n // 4
        remainder = n % 4  # To handle cases where n is not divisible by 4

        # Linear interpolation between each pair of diamond corners
        def interpolate(start, end, num_points):
            return np.linspace(start, end, num_points, endpoint=False)

        # Generate points for each edge
        top_right = interpolate(top, right,
                                points_per_edge + (1 if remainder > 0 else 0))
        remainder -= 1
        right_bottom = interpolate(right, bottom,
                                   points_per_edge + (1 if remainder > 0 else 0))
        remainder -= 1
        bottom_left = interpolate(bottom, left,
                                  points_per_edge + (1 if remainder > 0 else 0))
        remainder -= 1
        left_top = interpolate(left, top,
                               points_per_edge + (1 if remainder > 0 else 0))

        # Combine all the points into a single array
        diamond_points = np.vstack((top_right, right_bottom, bottom_left, left_top))
        kf = []
        for i in range(n):
            o = KandinskyUniverse.kandinskyShape()
            if t:
                o.color = random.choice(["pink", "green"])
                o.shape = random.choice(["diamond", "square"])
            else:
                o.color = random.choice(KandinskyUniverse.matplotlib_colors_list)
                o.shape = random.choice(KandinskyUniverse.kandinsky_shapes)
            o.size = so
            o.x = diamond_points[i, 0]
            o.y = diamond_points[i, 1]
            kf.append(o)

        random_percent = random.uniform(min_percent, max_percent)
        kf = kf[:int(len(kf) * random_percent)]
        return kf

    def _shapesOnShapes(self, truth):
        so = 0.04

        combis = random.randint(0, 2)
        if combis == 0:  g = lambda so, truth: self._bigCircle(so,
                                                               truth) + self._bigSquare(
            so, truth)
        if combis == 1:  g = lambda so, truth: self._bigCircle(so,
                                                               truth) + self._bigTriangle(
            so, truth)
        if combis == 2:  g = lambda so, truth: self._bigSquare(so,
                                                               truth) + self._bigTriangle(
            so, truth)

        kf = g(so, truth)
        t = 0
        tt = 0
        maxtry = 1000
        while KandinskyUniverse.overlaps(kf) and (t < maxtry):
            kf = g(so, truth)
            if tt > 10:
                tt = 0
                so = so * 0.90
            tt = tt + 1
            t = t + 1
        return kf

    def _only(self, truth, shape, size=None, lw=None):
        so = 0.05

        # bk basic patterns
        if shape == "random":
            g = lambda so, truth: self._random(size)
        elif shape == "circle":
            g = lambda so, truth: self._circle(size, lw)
        elif shape == "circle_group":
            g = lambda so, truth: self._bigCircle(so, truth)
        elif shape == "triangle":
            g = lambda so, truth: self._triangle(size, lw)
        elif shape == "triangle_group":
            g = lambda so, truth: self._bigTriangle(so, truth)
        elif shape == "triangle_group_cf":
            g = lambda so, truth: self._bigTriangleCF(so, truth)
        elif shape == "triangle_group_cf2":
            g = lambda so, truth: self._bigTriangleCF2(so, truth)
        elif shape == "square":
            g = lambda so, truth: self._square(size, lw)
        elif shape == "square_group":
            g = lambda so, truth: self._bigSquare(so, truth)
        elif shape == "square_group_cf":
            g = lambda so, truth: self._bigSquareCF(so, truth)
        elif shape == "square_group_cf2":
            g = lambda so, truth: self._bigSquareCF2(so, truth)
        elif shape == "diamond":
            g = lambda so, truth: self._bigDiamond(so, truth)

        # challenge patterns
        elif shape == "gestalt_triangle":
            g = lambda so, truth: self._gestaltTriangle(so, truth)
        elif shape == "gestalt_triangle_cf":
            g = lambda so, truth: self._gestaltTriangleCF(so, truth)
        elif shape == "gestalt_circle_triangle":
            g = lambda so, truth: self._gestaltCircleTriangle(so, truth)
        elif shape == "proximity_square":
            g = lambda so, truth: self._proximity_square(so, truth)
        elif shape == "proximity_square_cf":
            g = lambda so, truth: self._proximity_squareCF(so, truth)
        elif shape == "similarity_triangle_circle":
            g = lambda so, truth: self._similarity_triangle_circle()
        elif shape == "similarity_triangle_circle_cf":
            g = lambda so, truth: self._similarity_triangle_circleCF()
        elif shape == "continue_two_curves":
            g = lambda so, truth: self._continue_two_curves()
        elif shape == "continue_two_curves_cf":
            g = lambda so, truth: self._continue_two_curves_cf(so, truth)
        elif shape == "squarecircle":
            g = lambda so, truth: self._bigSquare(so, truth) + self._bigCircle(so,
                                                                               truth)


        elif shape == "diamondcircle":
            g = lambda so, truth: self._bigDiamond(so, truth) + self._bigCircle(so,
                                                                                truth)
        elif shape == "trianglecircle":
            g = lambda so, truth: self._bigCircle(so, truth) + self._bigTriangle(so,
                                                                                 truth)
        elif shape == "trianglecircle_flex":
            g = lambda so, truth: self._bigCircleFlex(so,
                                                      truth) + self._bigTriangler(so,
                                                                                  truth)
        elif shape == 'trianglesquare':
            g = lambda so, truth: self._bigSquare(so, truth) + self._bigTriangle(so, truth)
        elif shape == 'trianglesquare_cf':
            g = lambda so, truth: self._bigSquareCF(so, truth) + self._bigTriangleCF(so, truth)
        elif shape == "circlesquare_count":
            g = lambda so, truth: self._smallCircleFlex(
                so, truth) + self._smallSquare(so, truth) + self._smallCircleFlex(so,
                                                                                  truth)
        elif shape == "trianglesquarecircle":
            g = lambda so, truth: self._bigSquare(so, truth) + self._bigCircle(so,
                                                                               truth) + self._bigTriangle(
                so, truth)
        elif shape == "trianglepartsquare":
            g = lambda so, truth: self._bigSquare(so, truth, min_percent=0,
                                                  max_percent=1) + self._bigTriangle(
                so,
                truth)
        elif shape == "parttrianglepartsquare":
            g = lambda so, truth: self._bigSquare(
                so, truth, min_percent=0, max_percent=1) + self._bigTriangle(so,
                                                                             truth,
                                                                             min_percent=0,
                                                                             max_percent=0.8)
        else:
            raise ValueError("Shape does not support.")

        kf = g(so, truth)
        t = 0
        tt = 0
        maxtry = 1000
        while (KandinskyUniverse.overlaps(kf) or KandinskyUniverse.overflow(
                kf)) and (t < maxtry):
            kf = g(so, truth)
            if tt > 10:
                tt = 0
                so = so * 0.90
            tt = tt + 1
            t = t + 1
        return kf

    def tri_only(self, n=1, rule_style=False, size_lw=None):
        kfs = []
        if n > 0:
            for size, lw in size_lw:
                kf = self._only(rule_style, "triangle", size, lw)
                kfs.append(kf)
        return kfs

    def tri_group(self, n=1, rule_style=False):
        kfs = []
        if rule_style:
            for i in range(n):
                kf = self._only(rule_style, "triangle_group")
                kfs.append(kf)
        else:
            kfs.append(self._only(rule_style, "triangle_group_cf"))
            kfs.append(self._only(rule_style, "triangle_group_cf2"))
            kfs.append(self._only(rule_style, "triangle_group_cf2"))

        return kfs

    def gestalt_triangle(self, n=1, rule_style=False):
        kfs = []
        if rule_style:
            for i in range(n):
                kf = self._only(rule_style, "gestalt_triangle")
                kfs.append(kf)
        else:
            kfs.append(self._only(rule_style, "gestalt_triangle_cf"))
            kfs.append(self._only(rule_style, "gestalt_triangle_cf"))
            kfs.append(self._only(rule_style, "gestalt_triangle_cf"))
        return kfs

    def continue_two_curves(self, n=1, rule_style=False):
        kfs = []
        if rule_style:
            for i in range(n):
                kf = self._only(rule_style, "continue_two_curves")
                kfs.append(kf)
        else:
            kfs.append(self._only(rule_style, "continue_two_curves_cf"))
            kfs.append(self._only(rule_style, "continue_two_curves_cf"))
            kfs.append(self._only(rule_style, "continue_two_curves_cf"))
        return kfs

    def gestalt_circle_triangle(self, n=1, rule_style=False):
        kfs = []
        for i in range(n):
            kf = self._only(rule_style, "gestalt_circle_triangle")
            kfs.append(kf)
        return kfs

    def proximity_square(self, n=1, rule_style=False):
        kfs = []
        if rule_style:
            for i in range(n):
                kf = self._only(rule_style, "proximity_square")
                kfs.append(kf)
        else:
            kfs.append(self._only(rule_style, "proximity_square_cf"))
            kfs.append(self._only(rule_style, "proximity_square_cf"))
            kfs.append(self._only(rule_style, "proximity_square_cf"))
        return kfs

    def similarity_triangle_circle(self, n=1, rule_style=False):
        kfs = []
        if rule_style:
            for i in range(n):
                kf = self._only(rule_style, "similarity_triangle_circle")
                kfs.append(kf)
        else:
            kfs.append(self._only(rule_style, "similarity_triangle_circle_cf"))
            kfs.append(self._only(rule_style, "similarity_triangle_circle_cf"))
            kfs.append(self._only(rule_style, "similarity_triangle_circle_cf"))
        return kfs

    def cir_only(self, n=1, rule_style=False, size_lw=None):
        kfs = []
        if n > 0:
            for size, lw in size_lw:
                kf = self._only(rule_style, "circle", size, lw)
                kfs.append(kf)
        return kfs

    def cir_group(self, n=1, rule_style=False):
        kfs = []
        for i in range(n):
            kf = self._only(rule_style, "circle_group")
            kfs.append(kf)
        return kfs

    def dia_only(self, n=1, rule_style=False):
        kfs = []
        for i in range(n):
            # print(i)
            kf = self._only(True, "diamond_group")
            kfs.append(kf)
        return kfs

    def square_only(self, n=1, rule_style=False, size_lw=None):
        kfs = []
        if n > 0:
            for size, lw in size_lw:
                # print(i)
                kf = self._only(True, "square", size, lw)
                kfs.append(kf)
        return kfs

    def square_group(self, n=1, rule_style=False):
        kfs = []
        if rule_style:
            for i in range(n):
                # print(i)
                kf = self._only(True, "square_group")
                kfs.append(kf)
        else:
            kfs.append(self._only(True, "square_group_cf"))
            kfs.append(self._only(True, "square_group_cf2"))
        return kfs

    def triangle_circle(self, n=1, rule_style=False):
        kfs = []
        for i in range(n):
            kf = self._only(rule_style, "trianglecircle")
            kfs.append(kf)
        return kfs

    def triangle_circle_flex(self, n=1, rule_style=False):
        kfs = []
        for i in range(n):
            kf = self._only(rule_style, "trianglecircle_flex")
            kfs.append(kf)
        return kfs

    def triangle_circle_flex_cf(self, n=1, rule_style=False):
        kfs = []
        for i in range(n):
            kf = self._only(False, "trianglecircle_flex")
            kfs.append(kf)
        return kfs

    def circle_square_count(self, n=1, rule_style=False):
        kfs = []
        for i in range(n):
            kf = self._only(rule_style, "circlesquare_count")
            kfs.append(kf)
        return kfs

    def circle_square_count_cf(self, n=1, rule_style=False):
        kfs = []
        for i in range(n):
            kf = self._only(False, "circlesquare_count")
            kfs.append(kf)
        return kfs

    def diamond_circle_cf(self, n=1, rule_style=False):
        kfs = []
        for i in range(n):
            kf = self._only(False, "trianglecircle")
            kfs.append(kf)
        return kfs

    def square_circle(self, n=1, rule_style=False):
        kfs = []
        for i in range(n):
            kf = self._only(True, "squarecircle")
            kfs.append(kf)
        return kfs

    def triangle_square(self, n=1, rule_style=False):
        kfs = []
        if rule_style:
            for i in range(n):
                # print(i)
                kf = self._only(True, "trianglesquare")
                kfs.append(kf)
        else:
            kfs.append(self._only(True, "trianglesquare_cf"))
            kfs.append(self._only(True, "trianglesquare_cf"))
            kfs.append(self._only(True, "trianglesquare_cf"))

        return kfs

    def diamond_circle(self, n=1, rule_style=False):
        kfs = []
        for i in range(n):
            # print(i)
            kf = self._only(True, "diamondcircle")
            kfs.append(kf)
        return kfs

    def triangle_circle_cf(self, n=1, rule_style=False):
        kfs = []
        for i in range(n):
            kf = self._only(False, "trianglecircle")
            kfs.append(kf)
        return kfs

    def triangle_square_circle(self, n=1, rule_style=False):

        kfs = []
        for i in range(n):
            # print(i)
            kf = self._only(True, "trianglesquarecircle")
            kfs.append(kf)
        return kfs

    def triangle_partsquare(self, n=1, rule_style=False):
        kfs = []
        for i in range(n):
            # print(i)
            kf = self._only(True, "trianglepartsquare")
            kfs.append(kf)
        return kfs

    def parttriangle_partsquare(self, n=1, rule_style=False):
        kfs = []
        for i in range(n):
            # print(i)
            kf = self._only(True, "parttrianglepartsquare")
            kfs.append(kf)
        return kfs

    def true_kf(self, n=1, rule_style=False):
        kfs = []
        for i in range(n):
            kf = self._shapesOnShapes(True)
            kfs.append(kf)
        return kfs

    def almost_true_kf(self, n=1, rule_style=False):
        kfs = []

        for i in range(n):
            # print(i)
            kf = self._shapesOnShapes(False)
            kfs.append(kf)
        return kfs

    def false_kf(self, n=1, rule_style=False):
        # we are  sure that random image does not contain "shapes on shapes"
        t = self.min + self.max
        rg = Random(self.u, t, t)
        return rg.true_kf(n)

    def _random(self, size):
        # we are sure that random image does not contain "shapes on shapes"

        rg = Random(self.u, size, size)
        return rg.true_kf(1)
