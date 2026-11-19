import numpy as np
from core import Core
from colors import *
from math import cos, sin, radians

SCR_H = 240
SCR_W = 320

OFY = SCR_H // 2
OFX = SCR_W // 2

class GTE():
    def __init__(self, core: Core, fov=160):
        self._fov = fov
        self._core = core
        
        self._buffer = self._core._buffer
        self._grid_yx = self._core._grid_yx
        self._grid_y = self._core._grid_y
        self._grid_x = self._core._grid_x
        
    def edge_function(self, ax, ay, bx, by, px, py):
        return (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        
    def projection(self, point):
        x, y, z = point[..., 0], point[..., 1], point[..., 2]
        sx = self._core._scr_width // 2 + x * self._fov // z
        sy = self._core._scr_height // 2 - y * self._fov // z

        return np.stack((sx, sy), axis=-1).astype(int)

    def rotate_y_deg(self, angle):
        a = radians(angle)
        cos_a = cos(a)
        sin_a = sin(a)
        
        rot_mat = np.array([
            [cos_a,  0, sin_a],
            [0,      1, 0    ],
            [-sin_a, 0, cos_a],
        ])
                
        return rot_mat
        
    def rotate_x_deg(self, angle):
        a = radians(angle)
        cos_a = cos(a)
        sin_a = sin(a)
        
        rot_mat = np.array([
            [1,  0,      0    ],
            [0,  cos_a, -sin_a],
            [0,  sin_a,  cos_a],
        ])
                
        return rot_mat
        
    def rotate_z_deg(self, angle):
        a = radians(angle)
        cos_a = cos(a)
        sin_a = sin(a)
        
        rot_mat = np.array([
            [cos_a, -sin_a,  0    ],
            [sin_a,  cos_a,  0    ],
            [0,      0,      1    ],
        ])
                
        return rot_mat