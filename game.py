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

WALL = np.array([
    [[-h, -h, 0], [ h, -h, 0], [ h,  h, 0]],
    [[-h, -h, 0], [ h,  h, 0], [-h,  h, 0]],
])[:, ::-1]

UV_A = np.array(((0, 0), (1, 0), (1, 1)))
UV_B = np.array(((0, 0), (1, 1), (0, 1)))
FACE_UV = np.array([UV_A, UV_B])[:, ::-1]

CELL_WALL = 1
CELL_FLOOR = 0

def cell_center(row, col):
    return np.array([(col + 0.5) * CELL, 0, (row + 0.5) * CELL])

def is_empty(row, col):
    if row < 0 or row >= len(MAP) or col < 0 or col >= len(MAP[0]):
        return False        # outside the map counts as solid
    return MAP[row][col] == CELL_FLOOR

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

        self.positions = []
        for i, row in enumerate(MAP):
            for j, col in enumerate(row):
                wall_n = wall_e = wall_s = wall_w = False

                wall_n = is_empty(i - 1, j)
                wall_e = is_empty(i, j + 1)
                wall_s = is_empty(i + 1, j)
                wall_w = is_empty(i, j - 1)

                self.positions.append({
                    "position": cell_center(i, j),
                    "cell": col,
                    "walls": [wall_n, wall_e, wall_s, wall_w]
                })

        floor = FLOOR
        self.model_floor = floor
        self.world_floor = np.array([floor + face['position'] for face in self.positions if face['cell'] == CELL_FLOOR])
        self.colors = np.array(
            [[COLOR_LIGHT_GRAY for _ in range(3)] for _ in range(len(self.world_floor))], 
            dtype=np.float32
        )

        n = len(self.world_floor)
        self.uvs_all    = np.tile(self.uvs, (n, 1, 1, 1))                        # (faces, 2, 3, 2)
        self.colors_all = np.full((n, 2, 3, 3), COLOR_LIGHT_GRAY, dtype=np.float32)  # (faces, 2, 3, 3)

        wall = WALL
        self.model_wall = wall

        quads = []
        for face in self.positions:
            if face['cell'] != CELL_WALL:
                continue

            if face['walls'][0]:
                quads.append(wall + face['position'] + (0, 50, -50))

            if face['walls'][1]:
                quads.append(wall @ gte.rotate_y_deg(-90).T + face['position'] + (50, 50, 0))

            if face['walls'][2]:
                quads.append(wall @ gte.rotate_y_deg(180).T + face['position'] + (0, 50, 50))

            if face['walls'][3]:
                quads.append(wall @ gte.rotate_y_deg(90).T + face['position'] + (-50, 50, 0))
        
        self.world_wall = np.array(quads)
        n = len(self.world_wall)

        self.uvs_all = np.vstack((
            self.uvs_all,
            np.tile(self.uvs, (n, 1, 1, 1))
        ))
        self.colors_all = np.vstack((
            self.colors_all,
            np.full((n, 2, 3, 3), COLOR_WHITE, dtype=np.float32)
        ))

    def render(self, t, dt):
        self.update_camera(dt)
        self.render_state.buffer[...] = COLOR_YELLOW

        view = gte.rotate_x_deg(self.pitch) @ gte.rotate_y_deg(-self.yaw)
        tris_3d = np.vstack((
            (self.world_floor - self.cam_pos) @ view.T,
            (self.world_wall - self.cam_pos) @ view.T,
        ))

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