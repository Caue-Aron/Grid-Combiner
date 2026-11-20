import numpy as np
from PIL import Image
from random import choice
from colors import *
from gte import GTE
from core import Core
from gpu import GPU

SCR_H = 240
SCR_W = 320

OFY = SCR_H // 2
OFX = SCR_W // 2
H = 160

tri_3d = np.array([
    ((  0,  50, 0), ( 60, -40, 0), (-60, -40, 0)),
    ((  0,  50, 0), ( 50, -50, 0), (-50, -50, 0)),
    ((-80,  30, 0), (-20, -30, 0), (-90, -40, 0)),
])

s = 50
cube_verts = np.array([
    (-s, -s, -s), ( s, -s, -s), ( s,  s, -s), (-s,  s, -s),   # 0-3: near face (z = -s)
    (-s, -s,  s), ( s, -s,  s), ( s,  s,  s), (-s,  s,  s),   # 4-7: far face  (z = +s)
])

cube_idx = np.array([
    (0, 1, 2), (0, 2, 3),   # front
    (5, 4, 7), (5, 7, 6),   # back
    (4, 0, 3), (4, 3, 7),   # left
    (1, 5, 6), (1, 6, 2),   # right
    (3, 2, 6), (3, 6, 7),   # top
    (0, 4, 5), (0, 5, 1),   # bottom
])
cube_idx = cube_idx[:, ::-1]

tri_3d = cube_verts[cube_idx]

tri = np.array([[0, 50, 160], [60, -40, 320], [-60, -30, 140]])

if __name__ == '__main__':
    core = Core(SCR_W, SCR_H)
    gte = GTE(core, 160)
    gpu = GPU(core, gte)

    rot_x = gte.rotate_x_deg(-15)
    rot_y = gte.rotate_y_deg(45)
    rot_z = gte.rotate_z_deg(60)
    
    rot_mat = rot_y @ rot_x @ rot_z
    tri_3d = tri_3d @ rot_mat.T
    
    tri_3d = tri_3d + (0, 0, 200)
    
    tri = gte.projection(tri_3d[...])
    
    face_colors = [[colors[i], colors[i+1], colors[i+2]] for i in range(len(tri))]

    for t, c in zip(tri, face_colors):
        gpu.shaded_triangle(t, c)

    # tri = gte.projection(tri)
    # gpu.shaded_triangle(tri, (COLOR_MAGENTA, COLOR_GREEN, COLOR_BLUE))
    
    img = Image.fromarray(core._buffer)
    img.save('output.png')