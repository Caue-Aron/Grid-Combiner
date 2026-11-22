import numpy as np
from colors import *
from core import Core
from gte import GTE

class GPU:
    def __init__(self, core:Core, gte: GTE):
        self._core = core
        self._gte = gte
        
        self.clear_color = COLOR_YELLOW
        
        self._buffer = self._core._buffer
        self._grid_yx = self._core._grid_yx
        self._grid_y = self._core._grid_y
        self._grid_x = self._core._grid_x
        
        self.clear_scr()

    def rectangle(self, buffer, x, y, width, height):
        mask_x = (self._grid_x >= x) & (self._grid_x < x + width)
        mask_y = (self._grid_y >= y) & (self._grid_y < y + height)
        
        buffer[mask_y & mask_x] = COLOR_BLUE
        
    def triangle(self, tri, color):
        h, w = self._buffer.shape[:2]
        
        x_list = tri[:, 0]
        y_list = tri[:, 1]

        x_min = max(x_list.min(), 0)
        x_max = min(x_list.max(), w - 1)
        y_min = max(y_list.min(), 0)
        y_max = min(y_list.max(), h - 1)

        if x_min > x_max or y_min > y_max:
            return
        
        view_y, view_x = self._grid_yx[
            :,
            y_min:y_max + 1,
            x_min:x_max + 1
        ]
        
        eab_1 = GTE.edge_function(*tri[0], *tri[1], view_x, view_y)
        eab_2 = GTE.edge_function(*tri[1], *tri[2], view_x, view_y)
        eab_3 = GTE.edge_function(*tri[2], *tri[0], view_x, view_y)
        
        inside = (eab_1 >= 0) & (eab_2 >= 0) & (eab_3 >= 0)
        
        self._buffer[
            y_min:y_max + 1,
            x_min:x_max + 1
        ][inside] = color
        
    def shaded_triangle(self, tri, colors):
        area = GTE.edge_function(*tri[0], *tri[1], *tri[2])
        if area <= 0:
            return
        
        h, w = self._buffer.shape[:2]
        
        x_list = tri[:, 0]
        y_list = tri[:, 1]

        x_min = max(x_list.min(), 0)
        x_max = min(x_list.max(), w - 1)
        y_min = max(y_list.min(), 0)
        y_max = min(y_list.max(), h - 1)

        if x_min > x_max or y_min > y_max:
            return
        
        xs_box = np.arange(x_min, x_max + 1)            # one entry per column of the box
        ys_box = np.arange(y_min, y_max + 1)[:, None]   # one entry per row of the box

        A1, B1, C1 = GTE.edge_coeffs(*tri[0], *tri[1])
        A2, B2, C2 = GTE.edge_coeffs(*tri[1], *tri[2])
        A3, B3, C3 = GTE.edge_coeffs(*tri[2], *tri[0])

        e1 = A1 * xs_box + (B1 * ys_box + C1)
        e2 = A2 * xs_box + (B2 * ys_box + C2)
        e3 = A3 * xs_box + (B3 * ys_box + C3)

        inside = (e1 >= 0) & (e2 >= 0) & (e3 >= 0)

        ys, xs = np.nonzero(inside)
        xs_s = xs + x_min
        ys_s = ys + y_min

        a, b, c = GTE.plane_coeffs(tri, colors)

        values = a * xs_s[:, None] + b * ys_s[:, None] + c
        
        self._buffer[ys_s, xs_s] = values

    def textured_triangle(self, tri, uv, img):
        area = GTE.edge_function(*tri[0], *tri[1], *tri[2])
        if area <= 0:
            return
        
        h, w = self._buffer.shape[:2]
        
        x_list = tri[:, 0]
        y_list = tri[:, 1]

        x_min = max(x_list.min(), 0)
        x_max = min(x_list.max(), w - 1)
        y_min = max(y_list.min(), 0)
        y_max = min(y_list.max(), h - 1)

        if x_min > x_max or y_min > y_max:
            return
        
        xs_box = np.arange(x_min, x_max + 1)            # one entry per column of the box
        ys_box = np.arange(y_min, y_max + 1)[:, None]   # one entry per row of the box

        A1, B1, C1 = GTE.edge_coeffs(*tri[0], *tri[1])
        A2, B2, C2 = GTE.edge_coeffs(*tri[1], *tri[2])
        A3, B3, C3 = GTE.edge_coeffs(*tri[2], *tri[0])

        e1 = A1 * xs_box + (B1 * ys_box + C1)
        e2 = A2 * xs_box + (B2 * ys_box + C2)
        e3 = A3 * xs_box + (B3 * ys_box + C3)

        inside = (e1 >= 0) & (e2 >= 0) & (e3 >= 0)

        ys, xs = np.nonzero(inside)
        xs_s = xs + x_min
        ys_s = ys + y_min

        size = np.array(img.shape[:2][::-1]) - 1     # (W-1, H-1), the largest u and v in texels
        uv_texel = uv * (np.array(img.shape[:2][::-1]) - 1)   # TODO: scale the (3, 2) vertex UVs by size, once per triangle

        a, b, c = GTE.plane_coeffs(tri, uv_texel)   # each has shape (2,): one for u, one for v

        t = a * xs_s[:, None] + b * ys_s[:, None] + c
        t = t.astype(int)
        texel = img[t[:, 1], t[:, 0]]

        self._buffer[ys_s, xs_s] = texel

    def shaded_textured_triangle(self, tri, uv, img, colors):
        area = GTE.edge_function(*tri[0], *tri[1], *tri[2])
        if area <= 0:
            return
        
        h, w = self._buffer.shape[:2]
        
        x_list = tri[:, 0]
        y_list = tri[:, 1]

        x_min = max(x_list.min(), 0)
        x_max = min(x_list.max(), w - 1)
        y_min = max(y_list.min(), 0)
        y_max = min(y_list.max(), h - 1)

        if x_min > x_max or y_min > y_max:
            return
        
        xs_box = np.arange(x_min, x_max + 1)            # one entry per column of the box
        ys_box = np.arange(y_min, y_max + 1)[:, None]   # one entry per row of the box

        A1, B1, C1 = GTE.edge_coeffs(*tri[0], *tri[1])
        A2, B2, C2 = GTE.edge_coeffs(*tri[1], *tri[2])
        A3, B3, C3 = GTE.edge_coeffs(*tri[2], *tri[0])

        e1 = A1 * xs_box + (B1 * ys_box + C1)
        e2 = A2 * xs_box + (B2 * ys_box + C2)
        e3 = A3 * xs_box + (B3 * ys_box + C3)

        inside = (e1 >= 0) & (e2 >= 0) & (e3 >= 0)

        ys, xs = np.nonzero(inside)
        xs_s = xs + x_min
        ys_s = ys + y_min

        size = np.array(img.shape[:2][::-1]) - 1     # (W-1, H-1), the largest u and v in texels
        uv_texel = uv * size   # TODO: scale the (3, 2) vertex UVs by size, once per triangle

        a, b, c = GTE.plane_coeffs(tri, np.hstack([colors, uv_texel]))   # each has shape (2,): one for u, one for v

        ca, cb, cc = a[:3], b[:3], c[:3]
        ua, ub, uc = a[3:], b[3:], c[3:]

        t = ua * xs_s[:, None] + ub * ys_s[:, None] + uc
        t = t.astype(int)
        texel = img[t[:, 1], t[:, 0]]

        gouraud = ca * xs_s[:, None] + cb * ys_s[:, None] + cc

        self._buffer[ys_s, xs_s] = texel * gouraud // 255

        return True
        
    def set_clear_color(self, color):
        self.clear_color = color
        
    def clear_scr(self):
        self._buffer[...] = self.clear_color

if __name__ == "__main__":
    tri = np.array([[0, 0],
                    [4, 0],
                    [0, 4]])

    tri2 = tri + np.array([10, 20])       # same triangle, moved to start at (10, 20)

    A = np.array([[  0, 10],    # vertex 0: red, green
                [100, 90],    # vertex 1
                [200, 50]])   # vertex 2

    a, b, c = GTE.plane_coeffs(tri2, A)
    print(a, b, c)

    x_min, y_min = 10, 20
    ys_box, xs_box = np.mgrid[y_min:y_min + 5, x_min:x_min + 5]

    e1 = GTE.edge_function(*tri2[0], *tri2[1], xs_box, ys_box)
    e2 = GTE.edge_function(*tri2[1], *tri2[2], xs_box, ys_box)
    e3 = GTE.edge_function(*tri2[2], *tri2[0], xs_box, ys_box)
    inside = (e1 >= 0) & (e2 >= 0) & (e3 >= 0)

    ys, xs = np.nonzero(inside)
    print(inside.sum(), len(xs))
    print(xs, ys)

    xs_s = xs + x_min
    ys_s = ys + y_min

    out = a * xs_s[:, None] + b * ys_s[:, None] + c
    print(out.shape)
    print(out)

    canvas = np.zeros((5, 5))
    canvas[ys, xs] = out[:, 0]      # out[:, 0] is the red column
    print(canvas)

    x_min, y_min = 10, 20
    tri2 = tri + np.array([10, 20])

    xs_box = np.arange(x_min, x_min + 5)             # one entry per column
    ys_box = np.arange(y_min, y_min + 5)[:, None]    # one entry per row

    # new way: edge 1 (from tri2[0] to tri2[1])
    A1, B1, C1 = GTE.edge_coeffs(*tri2[0], *tri2[1])
    e1_new = A1 * xs_box + B1 * ys_box + C1

    # old way: your existing function on full 2D grids
    ys_grid, xs_grid = np.mgrid[y_min:y_min + 5, x_min:x_min + 5]
    e1_old = GTE.edge_function(*tri2[0], *tri2[1], xs_grid, ys_grid)

    print(e1_new.shape, e1_old.shape)
    print(np.array_equal(e1_new, e1_old))