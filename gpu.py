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
        
    def set_clear_color(self, color):
        self.clear_color = color
        
    def clear_scr(self):
        self._buffer[...] = self.clear_color