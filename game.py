import numpy as np
from PIL import Image
from random import choice
from colors import *
import gte
import gpu
from v_out import VOut
from math import radians, sin, cos
from map import MAP

CELL = 100
h = CELL / 2

# horizontal floor quad centered on the origin, at height y = 0
FLOOR = np.array([
    [[-h, 0, -h], [ h, 0, -h], [ h, 0,  h]],
    [[-h, 0, -h], [ h, 0,  h], [-h, 0,  h]],
])[:, ::-1]

UV_A = np.array(((0, 0), (1, 0), (1, 1)))
UV_B = np.array(((0, 0), (1, 1), (0, 1)))
FACE_UV = np.array([UV_A, UV_B])[:, ::-1]

def cell_center(row, col):
    return np.array([(col + 0.5) * CELL, 0, (row + 0.5) * CELL])

class App(VOut):
    def __init__(self, title='Demo', face_count=6):
        super().__init__(title)

        self.keys = set()
        self.cam_pos = np.array([150.0, 75.0, 150.0])
        self.yaw = 0.0     # degrees, positive = turn right
        self.pitch = 0.0   # degrees, positive = look up

        self.bind("<KeyPress>", lambda e: self.keys.add(e.keysym))
        self.bind("<KeyRelease>", lambda e: self.keys.discard(e.keysym))
        self.focus_set()

        wall_img = Image.open('image.png')
        self.wall_img_array = np.array(wall_img)
        
        self.uvs = FACE_UV

        self.faces = []

        offset = 100
        for i, row in enumerate(MAP):
            for j, col in enumerate(row):
                if col == 0:
                    self.faces.append({
                        "position": cell_center(i, j)
                    })

        quad = FLOOR
        self.model = quad
        self.world_faces = [quad + face['position'] for face in self.faces]
        self.colors = np.array(
            [[COLOR_LIGHT_GRAY for _ in range(3)] for _ in range(len(self.world_faces))], 
            dtype=np.float32
        )

        
        # self.tris_3d = self.world_faces @ gte.rotate_x_deg(90).T
        # self.tris_3d = self.tris_3d @ gte.scale_mat(2, 1, 2)
        # self.tris_3d = self.tris_3d + (0, -220, 320)

        n = len(self.world_faces)
        self.uvs_all    = np.tile(self.uvs, (n, 1, 1, 1))                        # (faces, 2, 3, 2)
        self.colors_all = np.full((n, 2, 3, 3), COLOR_LIGHT_GRAY, dtype=np.float32)  # (faces, 2, 3, 3)

        pass

    def render(self, t, dt):
        self.update_camera(dt)
        self.render_state.buffer[...] = COLOR_YELLOW

        view = gte.rotate_x_deg(self.pitch) @ gte.rotate_y_deg(-self.yaw)
        tris_3d = (self.world_faces - self.cam_pos) @ view.T

        NEAR = 1
        visible = (tris_3d[..., 2] >= NEAR).all(axis=-1)   # (faces, 2)
        visible_tris = tris_3d[visible]
        
        tri_2d = gte.projection(self.render_state, 160, visible_tris)   # (T, 3, 2)
        uvs    = self.uvs_all[visible]                                  # (T, 3, 2)
        cols   = self.colors_all[visible]                               # (T, 3, 3)

        for tri, uv, c in zip(tri_2d, uvs, cols):
            gpu.shaded_textured_triangle(self.render_state.buffer, tri, uv, self.wall_img_array, c)

        return self.render_state.buffer.astype(np.uint8)

    def update_camera(self, dt):
        turn_speed = 90.0    # degrees per second
        move_speed = 200.0   # units per second
        shift = 'Shift_L' in self.keys or 'Shift_R' in self.keys

        if 'Left' in self.keys:
            self.yaw -= turn_speed * dt
        if 'Right' in self.keys:
            self.yaw += turn_speed * dt

        if shift:
            if 'Up' in self.keys:
                self.pitch += turn_speed * dt
            if 'Down' in self.keys:
                self.pitch -= turn_speed * dt
            self.pitch = max(-80.0, min(80.0, self.pitch))
        else:
            a = radians(self.yaw)
            forward = np.array([sin(a), 0.0, cos(a)])
            if 'Up' in self.keys:
                self.cam_pos += forward * move_speed * dt
            if 'Down' in self.keys:
                self.cam_pos -= forward * move_speed * dt

if __name__ == '__main__':
    app = App()
    app.mainloop()