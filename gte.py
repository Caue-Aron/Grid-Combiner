import numpy as np
from core import RenderState
from colors import *
from math import cos, sin, radians
from numba import njit

SCR_W = 200
SCR_H = 150

OFY = SCR_H // 2
OFX = SCR_W // 2

@njit(cache=True)
def edge_function(ax, ay, bx, by, px, py):
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)

@njit(cache=True)
def projection(render_state, H, point):
    n, m = point.shape[0], point.shape[1]
    cx = render_state.width // 2
    cy = render_state.height // 2
    out = np.empty((n, m, 2), np.int64)

    for i in range(n):
        for j in range(m):
            z = point[i, j, 2]
            out[i, j, 0] = cx + np.int64(np.floor(point[i, j, 0] * H / z))
            out[i, j, 1] = cy - np.int64(np.floor(point[i, j, 1] * H / z))

    return out

def rotate_y_deg(angle):
    a = radians(angle)
    cos_a = cos(a)
    sin_a = sin(a)
    
    rot_mat = np.array([
        [cos_a,  0, sin_a],
        [0,      1, 0    ],
        [-sin_a, 0, cos_a],
    ])
            
    return rot_mat

def scale_mat(sx, sy, sz):
    scale_mat = np.array([
        [sx, 0,  0 ],
        [0,  sy, 0 ],
        [0,  0,  sz],
    ])

    return scale_mat
    
def rotate_x_deg(angle):
    a = radians(angle)
    cos_a = cos(a)
    sin_a = sin(a)
    
    rot_mat = np.array([
        [1,  0,      0    ],
        [0,  cos_a, -sin_a],
        [0,  sin_a,  cos_a],
    ])
            
    return rot_mat
    
def rotate_z_deg(angle):
    a = radians(angle)
    cos_a = cos(a)
    sin_a = sin(a)
    
    rot_mat = np.array([
        [cos_a, -sin_a,  0    ],
        [sin_a,  cos_a,  0    ],
        [0,      0,      1    ],
    ])
            
    return rot_mat

@njit(cache=True)
def plane_coeffs(tri, A):
    # tri: array of shape (3, 2), the vertices [[x0, y0], [x1, y1], [x2, y2]]
    # A:   array of shape (3,), the attribute value at each vertex
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

@njit(cache=True)
def edge_coeffs(ax, ay, bx, by):
    A = -(by - ay)
    B = (bx - ax)
    C = -ay*(bx - ax) + ax*(by - ay)

    return A, B, C

if __name__ == '__main__':
    print(edge_function(*[-50, -50], *[ 50, -50], *[ 50,  50]))

    tri = np.array([[0, 0],
                    [4, 0],
                    [0, 4]])

    tri2 = tri + np.array([10, 20])       # same triangle, moved to start at (10, 20)

    A = np.array([[  0, 10],    # vertex 0: red, green
                [100, 90],    # vertex 1
                [200, 50]])   # vertex 2
    print(plane_coeffs(tri2, A))