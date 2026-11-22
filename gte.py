import numpy as np
from core import Core
from colors import *
from math import cos, sin, radians
from numba import njit

SCR_W = 200
SCR_H = 150

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
    
    def edge_function(ax, ay, bx, by, px, py):
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

    def scale_mat(self, sx, sy, sz):
        scale_mat = np.array([
            [sx, 0,  0 ],
            [0,  sy, 0 ],
            [0,  0,  sz],
        ])

        return scale_mat
        
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

    def plane_coeffs(tri, A):
        # tri: array of shape (3, 2), the vertices [[x0, y0], [x1, y1], [x2, y2]]
        # A:   array of shape (3,), the attribute value at each vertex
        A = np.array(A)

        x0, y0 = tri[0]
        x1, y1 = tri[1]
        x2, y2 = tri[2]

        dx1, dy1 = x1 - x0, y1 - y0
        dx2, dy2 = x2 - x0, y2 - y0
        det = dx1 * dy2 - dx2 * dy1

        d1 = A[1] - A[0]  # how much A changes along edge 1
        d2 = A[2] - A[0]  # how much A changes along edge 2

        a = (d1 * dy2 - d2 * dy1) / det
        b = (d2 * dx1 - d1 * dx2) / det
        c = A[0] - a * x0 - b * y0

        return a, b, c

    def edge_coeffs(ax, ay, bx, by):
        A = -(by - ay)
        B = (bx - ax)
        C = -ay*(bx - ax) + ax*(by - ay)

        return A, B, C

if __name__ == '__main__':
    print(GTE.edge_coeffs(0, 0, 4, 0))