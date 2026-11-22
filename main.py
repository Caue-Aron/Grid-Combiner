import numpy as np
from PIL import Image
from random import choice
from colors import *
from gte import GTE
from core import Core
from gpu import GPU
from v_out import VOut
from math import radians, sin, cos

SCR_W = 320
SCR_H = 240

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

from random import choice, uniform

class App(VOut):
    def __init__(self, gpu, title='Demo', cube_count=10):
        super().__init__(gpu, title)

        self.model = CUBE_TRIS[:, ::-1]
        self.uvs = CUBE_UVS

        self.cubes = []

        self.colors = [
                    [choice(colors) for _ in range(3)]
                    for _ in range(len(self.model))
                ]

        for _ in range(cube_count):
            self.cubes.append({
                "position": (
                    uniform(-100, 100),
                    uniform(-100, 100),
                    uniform(100, 300)
                ),
                "rotation": (
                    uniform(0, 360),
                    uniform(0, 360)
                ),
                "speed": (
                    uniform(10, 60),
                    uniform(10, 60)
                ),
                "colors": [
                    [choice(colors) for _ in range(3)]
                    for _ in range(len(self.model))
                ]
            })

    def render(self, t, dt):
        self._gpu.clear_scr()

        # rot = self._gte.rotate_y_deg(15 * t) @ self._gte.rotate_z_deg(30 * t)
        # big_tri = self.model @ rot.T
        # big_tri = big_tri + (0, 0, 110)
        # big_tri = self._gte.projection(big_tri)


        # self.tri_calls = 0

        # for tri, c, uv in zip(
        #     big_tri,
        #     self.colors,
        #     self.uvs
        # ):
            
        #     # self.tri_calls += self._gpu.shaded_triangle(
        #     #                 tri, uv, img_array, c
        #     #             ) or 0

        #     # self.tri_calls += self._gpu.shaded_triangle(
        #     #                 tri, c
        #     #             ) or 0

        #     self.tri_calls += self._gpu.shaded_textured_triangle(
        #                     tri, uv, img_array, c
        #                 ) or 0


        for cube in self.cubes:
            rx = cube["rotation"][0] + cube["speed"][0] * t
            rz = cube["rotation"][1] + cube["speed"][1] * t

            rot = (
                self._gte.rotate_y_deg(rx)
                @ self._gte.rotate_z_deg(rz)
            )

            tri_3d = self.model @ rot.T
            tri_3d = tri_3d + cube["position"]

            tri_2d = self._gte.projection(tri_3d)

            for tri, c, uv in zip(
                tri_2d,
                cube["colors"],
                self.uvs
            ):
                self._gpu.shaded_textured_triangle(
                    tri, uv, img_array, c
                )
            

        return self._core._buffer.astype(np.uint8)

if __name__ == '__main__':
    img = Image.open('image.png')
    img_array = np.array(img)

    core = Core(SCR_W, SCR_H)
    gte = GTE(core, 160)
    gpu = GPU(core, gte)
    
    img = Image.fromarray(core._buffer)
    img.save('output.png')

    app = App(gpu)
    app.mainloop()