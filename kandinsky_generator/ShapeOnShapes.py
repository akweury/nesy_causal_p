import random
import math
from tqdm import tqdm

from kandinsky_generator.src.kp.KandinskyTruth import KandinskyTruthInterfce
from kandinsky_generator.src.kp.KandinskyUniverse import kandinskyShape, overlaps
from kandinsky_generator.src.kp.RandomKandinskyFigure import Random


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
            o = kandinskyShape()
            d = i * 2 * math.pi / n
            if t:
                o.color = random.choice(["blue", "yellow"])
                o.shape = random.choice(["square", "triangle"])
            else:
                o.color = random.choice(self.u.kandinsky_colors)
                o.shape = random.choice(self.u.kandinsky_shapes)

            o.size = so
            o.x = x + r * math.cos(d)
            o.y = y + r * math.sin(d)
            kf.append(o)
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
            o = kandinskyShape()
            if t:
                o.color = random.choice(["yellow", "red"])
                o.shape = random.choice(["circle", "square"])
            else:
                o.color = random.choice(self.u.kandinsky_colors)
                o.shape = random.choice(self.u.kandinsky_shapes)
            o.size = so
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
            o = kandinskyShape()
            if t:
                o.color = random.choice(["yellow", "red"])
                o.shape = random.choice(["circle", "square"])
            else:
                o.color = random.choice(self.u.kandinsky_colors)
                o.shape = random.choice(self.u.kandinsky_shapes)
            o.size = so
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
            o = kandinskyShape()
            if t:
                o.color = random.choice(["yellow", "red"])
                o.shape = random.choice(["circle", "square"])
            else:
                o.color = random.choice(self.u.kandinsky_colors)
                o.shape = random.choice(self.u.kandinsky_shapes)
            o.size = so
            o.x = xs + (i + 1) * dxi
            o.y = ys + (i + 1) * dyi
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
            o = kandinskyShape()
            if t:
                o.color = random.choice(["blue", "red"])
                o.shape = random.choice(["circle", "triangle"])
            else:
                o.color = random.choice(self.u.kandinsky_colors)
                o.shape = random.choice(self.u.kandinsky_shapes)
            o.size = so
            o.x = minx + i * dx
            o.y = miny
            kf.append(o)
            o = kandinskyShape()
            if t:
                o.color = random.choice(["blue", "red"])
                o.shape = random.choice(["circle", "triangle"])
            else:
                o.color = random.choice(self.u.kandinsky_colors)
                o.shape = random.choice(self.u.kandinsky_shapes)
            o.size = so
            o.x = minx + i * dx
            o.y = maxy
            kf.append(o)

        for i in range(n - 1):
            o = kandinskyShape()
            if t:
                o.color = random.choice(["blue", "red"])
                o.shape = random.choice(["circle", "triangle"])
            else:
                o.color = random.choice(self.u.kandinsky_colors)
                o.shape = random.choice(self.u.kandinsky_shapes)
            o.size = so
            o.x = minx
            o.y = miny + (i + 1) * dx
            kf.append(o)
            o = kandinskyShape()
            if t:
                o.color = random.choice(["blue", "red"])
                o.shape = random.choice(["circle", "triangle"])
            else:
                o.color = random.choice(self.u.kandinsky_colors)
                o.shape = random.choice(self.u.kandinsky_shapes)
            o.size = so
            o.x = maxx
            o.y = miny + (i + 1) * dx
            kf.append(o)
        random_percent = random.uniform(min_percent, max_percent)
        kf = kf[:int(len(kf) * random_percent)]
        return kf

    def _shapesOnShapes(self, truth):
        so = 0.04

        combis = random.randint(0, 2)
        if combis == 0:  g = lambda so, truth: self._bigCircle(so, truth) + self._bigSquare(so, truth)
        if combis == 1:  g = lambda so, truth: self._bigCircle(so, truth) + self._bigTriangle(so, truth)
        if combis == 2:  g = lambda so, truth: self._bigSquare(so, truth) + self._bigTriangle(so, truth)

        kf = g(so, truth)
        t = 0
        tt = 0
        maxtry = 1000
        while overlaps(kf) and (t < maxtry):
            kf = g(so, truth)
            if tt > 10:
                tt = 0
                so = so * 0.90
            tt = tt + 1
            t = t + 1
        return kf

    def _only(self, truth, shape):
        so = 0.04
        if shape == "circle":
            g = lambda so, truth: self._bigCircle(so, truth)
        elif shape == "triangle":
            g = lambda so, truth: self._bigTriangle(so, truth)
        elif shape == "square":
            g = lambda so, truth: self._bigSquare(so, truth)
        elif shape == "trianglecircle":
            g = lambda so, truth: self._bigCircle(so, truth) + self._bigTriangle(so, truth)
        elif shape == 'trianglesquare':
            g = lambda so, truth: self._bigSquare(so, truth) + self._bigTriangle(so, truth)
        elif shape == "squarecircle":
            g = lambda so, truth: self._bigSquare(so, truth) + self._bigCircle(so, truth)
        elif shape == "trianglesquarecircle":
            g = lambda so, truth: self._bigSquare(so, truth) + self._bigCircle(so, truth) + self._bigTriangle(so, truth)
        elif shape == "trianglepartsquare":
            g = lambda so, truth: self._bigSquare(so, truth, min_percent=0, max_percent=1) + self._bigTriangle(so,
                                                                                                               truth)
        elif shape == "parttrianglepartsquare":
            g = lambda so, truth: self._bigSquare(
                so, truth, min_percent=0, max_percent=1) + self._bigTriangle(so, truth, min_percent=0, max_percent=0.8)
        else:
            raise ValueError("Shape must be either 'circle', 'triangle' or 'square'")
        kf = g(so, truth)
        t = 0
        tt = 0
        maxtry = 1000
        while overlaps(kf) and (t < maxtry):
            kf = g(so, truth)
            if tt > 10:
                tt = 0
                so = so * 0.90
            tt = tt + 1
            t = t + 1
        return kf

    def cir_only(self, n=1):
        print("MAKE CIRCLE")
        kfs = []
        for i in tqdm(range(n), desc="generating objects"):
            # print(i)
            kf = self._only(True, "circle")
            kfs.append(kf)
        return kfs

    def tri_only(self, n=1):
        print("MAKE TRIANGLE")
        kfs = []
        for i in tqdm(range(n), desc="generating objects"):
            # print(i)
            kf = self._only(True, "triangle")
            kfs.append(kf)
        return kfs

    def square_only(self, n=1):
        print("MAKE SQUARE")
        kfs = []
        for i in tqdm(range(n), desc="generating objects"):
            # print(i)
            kf = self._only(True, "square")
            kfs.append(kf)
        return kfs

    def triangle_circle(self, n=1):
        print("MAKE TRIANGLE AND CIRCLE")
        kfs = []
        for i in tqdm(range(n), desc="generating objects"):
            # print(i)
            kf = self._only(True, "trianglecircle")
            kfs.append(kf)
        return kfs

    def square_circle(self, n=1):
        print("MAKE SQUARE AND CIRCLE")
        kfs = []
        for i in tqdm(range(n), desc="generating objects"):
            # print(i)
            kf = self._only(True, "squarecircle")
            kfs.append(kf)
        return kfs

    def triangle_square(self, n=1):
        print("MAKE TRIANGLE AND SQUARE")
        kfs = []
        for i in tqdm(range(n), desc="generating objects"):
            # print(i)
            kf = self._only(True, "trianglesquare")
            kfs.append(kf)
        return kfs

    def triangle_square_circle(self, n=1):
        print("MAKE TRIANGLE AND SQUARE AND CIRCLE")
        kfs = []
        for i in tqdm(range(n), desc="generating objects"):
            # print(i)
            kf = self._only(True, "trianglesquarecircle")
            kfs.append(kf)
        return kfs

    def triangle_partsquare(self, n=1):
        print("MAKE TRIANGLE AND PART SQUARE")
        kfs = []
        for i in tqdm(range(n), desc="generating objects"):
            # print(i)
            kf = self._only(True, "trianglepartsquare")
            kfs.append(kf)
        return kfs

    def parttriangle_partsquare(self, n=1):
        print("MAKE PART TRIANGLE AND PART SQUARE")
        kfs = []
        for i in tqdm(range(n), desc="generating objects"):
            # print(i)
            kf = self._only(True, "parttrianglepartsquare")
            kfs.append(kf)
        return kfs

    def true_kf(self, n=1):
        print("MAKE TRUE")
        kfs = []
        for i in tqdm(range(n), desc="generating objects"):
            # print(i)
            kf = self._shapesOnShapes(True)
            kfs.append(kf)
        return kfs

    def almost_true_kf(self, n=1):
        kfs = []
        print("MAKE CONTRAFACTUALS")
        for i in tqdm(range(n), desc="generating objects"):
            # print(i)
            kf = self._shapesOnShapes(False)
            kfs.append(kf)
        return kfs

    def false_kf(self, n=1):
        print("MAKE FALSE")
        # we are  sure that random image does not contain "shapes on shapes"
        t = self.min + self.max
        rg = Random(self.u, t, t)
        return rg.true_kf(n)
