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
        
    def triangle(self, p1, p2, p3, color):
        x_list = list([p1[0], p2[0], p3[0]])
        y_list = list([p1[1], p2[1], p3[1]])
        
        view_y, view_x = self._grid_yx[
            :,
            min(y_list):max(y_list) + 1,
            min(x_list):max(x_list) + 1
        ]
        
        eab_1 = self._gte.edge_function(*p1, *p2, view_x, view_y)
        eab_2 = self._gte.edge_function(*p2, *p3, view_x, view_y)
        eab_3 = self._gte.edge_function(*p3, *p1, view_x, view_y)
        
        inside = (eab_1 >= 0) & (eab_2 >= 0) & (eab_3 >= 0)
        
        self._buffer[
            min(y_list):max(y_list) + 1,
            min(x_list):max(x_list) + 1
        ][inside] = color
        
        pass
        
    def set_clear_color(self, color):
        self.clear_color = color
        
    def clear_scr(self):
        self._buffer[...] = self.clear_color