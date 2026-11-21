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
        
        eab_1 = self._gte.edge_function(*tri[0], *tri[1], view_x, view_y)
        eab_2 = self._gte.edge_function(*tri[1], *tri[2], view_x, view_y)
        eab_3 = self._gte.edge_function(*tri[2], *tri[0], view_x, view_y)
        
        inside = (eab_1 >= 0) & (eab_2 >= 0) & (eab_3 >= 0)
        
        self._buffer[
            y_min:y_max + 1,
            x_min:x_max + 1
        ][inside] = color
        
    def shaded_triangle(self, tri, colors):
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
        
        eab_1 = self._gte.edge_function(*tri[0], *tri[1], view_x, view_y)
        eab_2 = self._gte.edge_function(*tri[1], *tri[2], view_x, view_y)
        eab_3 = self._gte.edge_function(*tri[2], *tri[0], view_x, view_y)
        
        inside = (eab_1 >= 0) & (eab_2 >= 0) & (eab_3 >= 0)

        double_area = eab_1 + eab_2 + eab_3

        if double_area.any() <= 0:
            raise Exception('Degenrate triangle')

        lbda_1 = eab_2 / double_area
        lbda_2 = eab_3 / double_area
        lbda_3 = eab_1 / double_area

        r_factor = lbda_1[..., np.newaxis] * colors[0]
        y_factor = lbda_2[..., np.newaxis] * colors[1]
        b_factor = lbda_3[..., np.newaxis] * colors[2]

        gouraud = r_factor + y_factor + b_factor
        gouraud = gouraud.astype(int)
        
        self._buffer[
            y_min:y_max + 1,
            x_min:x_max + 1
        ][inside] = gouraud[inside]

    def shaded_textured_triangle(self, tri, uv, img, colors):
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
        
        eab_1 = self._gte.edge_function(*tri[0], *tri[1], view_x, view_y)
        eab_2 = self._gte.edge_function(*tri[1], *tri[2], view_x, view_y)
        eab_3 = self._gte.edge_function(*tri[2], *tri[0], view_x, view_y)
        
        inside = (eab_1 >= 0) & (eab_2 >= 0) & (eab_3 >= 0)

        double_area = eab_1 + eab_2 + eab_3

        if double_area.any() <= 0:
            raise Exception('Degenrate triangle')

        lbda_1 = eab_2 / double_area
        lbda_2 = eab_3 / double_area
        lbda_3 = eab_1 / double_area

        p1_f = lbda_1[..., np.newaxis] * uv[0] * (np.array(img.shape[:2][::-1]) - 1)
        p2_f = lbda_2[..., np.newaxis] * uv[1] * (np.array(img.shape[:2][::-1]) - 1)
        p3_f = lbda_3[..., np.newaxis] * uv[2] * (np.array(img.shape[:2][::-1]) - 1)

        r_factor = lbda_1[..., np.newaxis] * colors[0]
        y_factor = lbda_2[..., np.newaxis] * colors[1]
        b_factor = lbda_3[..., np.newaxis] * colors[2]

        gouraud = r_factor + y_factor + b_factor
        gouraud = gouraud.astype(int)

        texture = (p1_f + p2_f + p3_f)
        texture = texture.astype(int)

        t = texture[inside]
        texel = img[t[:, 1], t[:, 0]]

        out = texel * gouraud[inside] // 255
        
        self._buffer[
            y_min:y_max + 1,
            x_min:x_max + 1
        ][inside] = np.clip(out, 0, 255)

    def textured_triangle(self, tri, uv, img):
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
        
        eab_1 = self._gte.edge_function(*tri[0], *tri[1], view_x, view_y)
        eab_2 = self._gte.edge_function(*tri[1], *tri[2], view_x, view_y)
        eab_3 = self._gte.edge_function(*tri[2], *tri[0], view_x, view_y)
        
        inside = (eab_1 >= 0) & (eab_2 >= 0) & (eab_3 >= 0)

        double_area = eab_1 + eab_2 + eab_3

        if double_area.any() <= 0:
            raise Exception('Degenrate triangle')

        lbda_1 = eab_2 / double_area
        lbda_2 = eab_3 / double_area
        lbda_3 = eab_1 / double_area

        p2_f = lbda_2[..., np.newaxis] * uv[1] * (np.array(img.shape[:2][::-1]) - 1)
        p3_f = lbda_3[..., np.newaxis] * uv[2] * (np.array(img.shape[:2][::-1]) - 1)
        p1_f = lbda_1[..., np.newaxis] * uv[0] * (np.array(img.shape[:2][::-1]) - 1)

        texture = (p1_f + p2_f + p3_f)
        texture = texture.astype(int)

        t = texture[inside]
        texel = img[t[:, 1], t[:, 0]]
        
        self._buffer[
            y_min:y_max + 1,
            x_min:x_max + 1
        ][inside] = texel
        
    def set_clear_color(self, color):
        self.clear_color = color
        
    def clear_scr(self):
        self._buffer[...] = self.clear_color