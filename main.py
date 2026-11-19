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

tri_3d = (
    ((  0,  50, 0), ( 60, -40, 0), (-60, -40, 0)),
    ((  0,  50, 0), ( 50, -50, 0), (-50, -50, 0)),
    ((-80,  30, 0), (-20, -30, 0), (-90, -40, 0)),
)

if __name__ == '__main__':
    core = Core(SCR_W, SCR_H)
    gte = GTE(core, 160)
    gpu = GPU(core, gte)

    tri_3d = [tuple(gte.rotate_y_deg(45, tri)) for tri in tri_3d]
    tri_3d = [tuple(gte.rotate_z_deg(45, tri)) for tri in tri_3d]
    tri_3d = [tuple(gte.rotate_x_deg(-15, tri)) for tri in tri_3d]
    
    tri_3d = [
        # [(x + 80, y + 60, z + 200) for x, y, z in triangle]
        [(x, y, z + 200) for x, y, z in triangle]
        for triangle in tri_3d
    ]
    
    tri = [
        tuple(gte.projection(*point) for point in triangle)
        for triangle in tri_3d
    ]
    
    print(gte.projection(80, 60, 200))
    
    tri = [
        [[x.astype(int) for x in point] for point in triangle] for triangle in tri
    ]
    
    for tri in tri:
        gpu.triangle(*tri, choice(colors))
    
    img = Image.fromarray(core._buffer)
    img.save('output.png')