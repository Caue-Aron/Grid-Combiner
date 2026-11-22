import numpy as np
from PIL import Image
from random import choice
from colors import *
import gte
from core import RenderState
import gpu
from v_out import VOut
from math import radians, sin, cos
from numba import typeof

SCR_W = 200
SCR_H = 150

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
UV_A = np.array(((0, 0), (1, 0), (1, 1)))
UV_B = np.array(((0, 0), (1, 1), (0, 1)))
CUBE_UVS = [UV_A, UV_B] * 6
CUBE_UVS = CUBE_UVS[::-1]
CUBE_TRIS = CUBE_TRIS - np.array([0, 0, 250])


tri_3d = CUBE_TRIS[:, ::-1]

tri = np.array([[0, 50, 160], [60, -40, 320], [-60, -30, 140]])

from random import choice, uniform

class App(VOut):
    def __init__(self, title='Demo', cube_count=70):
        super().__init__(title)

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
                "colors": np.array(
                    [
                        [choice(colors) for _ in range(3)]
                        for _ in range(len(self.model))
                    ],
                    dtype=np.float32,
                )
            })

    def render(self, t, dt):
        self.render_state.buffer[...] = COLOR_YELLOW

        for cube in self.cubes:
            rx = cube["rotation"][0] + cube["speed"][0] * t
            rz = cube["rotation"][1] + cube["speed"][1] * t

            rot = (
                gte.rotate_y_deg(rx)
                @ gte.rotate_z_deg(rz)
            )

            tri_3d = self.model @ rot.T
            tri_3d = tri_3d + cube["position"]

            tri_2d = gte.projection(self.render_state, 160, tri_3d)

            for tri, c, uv in zip(
                tri_2d,
                cube["colors"],
                self.uvs
            ):
                gpu.shaded_textured_triangle(self.render_state.buffer, tri, uv, img_array, c)
            

        return self.render_state.buffer

if __name__ == '__main__':
    img = Image.open('image.png')
    img_array = np.array(img)

    app = App(gpu)
    app.mainloop()