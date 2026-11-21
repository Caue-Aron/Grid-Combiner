import numpy as np
from PIL import Image
from random import choice
from colors import *
from gte import GTE
from core import Core
from gpu import GPU
from v_out import VOut
from math import radians, sin, cos

SCR_H = 240
SCR_W = 320

OFY = SCR_H // 2
OFX = SCR_W // 2

CUBE_TRIS = np.array([
    # near (z = 200)
    [[-50, -50, 200], [ 50, -50, 200], [ 50,  50, 200]],
    [[-50, -50, 200], [ 50,  50, 200], [-50,  50, 200]],
    # far (z = 300)
    [[ 50, -50, 300], [-50, -50, 300], [-50,  50, 300]],
    [[ 50, -50, 300], [-50,  50, 300], [ 50,  50, 300]],
    # left (x = -50)
    [[-50, -50, 300], [-50, -50, 200], [-50,  50, 200]],
    [[-50, -50, 300], [-50,  50, 200], [-50,  50, 300]],
    # right (x = 50)
    [[ 50, -50, 200], [ 50, -50, 300], [ 50,  50, 300]],
    [[ 50, -50, 200], [ 50,  50, 300], [ 50,  50, 200]],
    # top (y = -50)
    [[-50, -50, 300], [ 50, -50, 300], [ 50, -50, 200]],
    [[-50, -50, 300], [ 50, -50, 200], [-50, -50, 200]],
    # bottom (y = 50)
    [[-50,  50, 200], [ 50,  50, 200], [ 50,  50, 300]],
    [[-50,  50, 200], [ 50,  50, 300], [-50,  50, 300]],
])

# UVs, same pattern for every face: first tri of a quad, then second tri
UV_A = ((0, 0), (1, 0), (1, 1))
UV_B = ((0, 0), (1, 1), (0, 1))
CUBE_UVS = [UV_A, UV_B] * 6
CUBE_UVS = CUBE_UVS[::-1]
CUBE_TRIS = CUBE_TRIS - np.array([0, 0, 250])


tri_3d = CUBE_TRIS[:, ::-1]

tri = np.array([[0, 50, 160], [60, -40, 320], [-60, -30, 140]])

class App(VOut):
    def __init__(self, gpu, title='Demo'):
        super().__init__(gpu, title)
        self.model = CUBE_TRIS[:, ::-1]          # never modified
        self.face_colors = [
            [choice(colors) for _ in range(3)]
            for _ in range(len(self.model))
        ]

    def render(self, t, dt):
        self._gpu.clear_scr()

        rot = self._gte.rotate_y_deg(45 * t) @ self._gte.rotate_z_deg(30 * t)

        tri_3d = self.model @ rot.T              # rotate about own center
        tri_3d = tri_3d + (0, 0, 160)            # then place in the scene
        tri_2d = self._gte.projection(tri_3d)

        for tri, c, uv in zip(tri_2d, self.face_colors, CUBE_UVS):
            self._gpu.shaded_textured_triangle(tri, uv, img_array, c)

        return self._core._buffer

if __name__ == '__main__':
    img = Image.open('image.png')
    img_array = np.array(img)

    core = Core(SCR_W, SCR_H)
    gte = GTE(core, 160)
    gpu = GPU(core, gte)

    rot_x = gte.rotate_x_deg(-15)
    rot_y = gte.rotate_y_deg(45)
    rot_z = gte.rotate_z_deg(45)
    
    rot_mat = rot_y @ rot_z
    tri_3d = tri_3d @ rot_mat.T
    tri_3d = tri_3d + (0, 0, 160)
    
    tri = gte.projection(tri_3d[...])
    
    face_colors = [
        [choice(colors) for _ in range(3)]
        for _ in range(len(tri))
    ]

    for t, c, uv in zip(tri, face_colors, CUBE_UVS):
        gpu.shaded_textured_triangle(t, uv, img_array, c)

    # tri = gte.projection(tri)
    # gpu.shaded_textured_triangle(tri, ((.5, 1), (1, 0), (0, 0)), img_array, (COLOR_BLUE, COLOR_GREEN, COLOR_RED))
    
    img = Image.fromarray(core._buffer)
    img.save('output.png')

    app = App(gpu)
    app.mainloop()