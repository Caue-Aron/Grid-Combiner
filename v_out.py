import gpu
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import time
from core import RenderState

class VOut(tk.Tk):
    def __init__(self, title='Demo'):
        super().__init__()
        self.render_state = RenderState(
            buffer=np.zeros((240, 320, 3), np.uint8),
            width=320,
            height=240,
        )

        self.title(title)
        self.scale = 3
        self.geometry(f'{self.render_state.width*self.scale}x{self.render_state.height*self.scale}')

        self.fps, self.total = 30, 60
        self.frame_i = 0
        self.after(0, self.step)
        self.last = 0
        self.last = time.perf_counter()
        self.label = tk.Label(self)
        self.label.pack()

        self.time_ms = 0

    def render(self, t, dt):
        # your software renderer goes here; return an (H, W, 3) uint8 array
        return np.empty((self.render_state.width, self.render_state.height, 3), dtype=np.uint8)

    def step(self):
        now = time.perf_counter()
        dt = now - self.last
        self.last = now
        
        t = self.frame_i / self.fps
        t0 = time.perf_counter()
        frame = self.render(t, dt)
        ms = (time.perf_counter() - t0) * 1000
        self.time_ms += ms

        self.title(f'Demo - {ms:.2f}')

        img = Image.fromarray(frame).resize((self.render_state.width*self.scale, self.render_state.height*self.scale), Image.NEAREST)
        self.photo = ImageTk.PhotoImage(img)   # keep a reference on self
        self.label.config(image=self.photo)

        self.frame_i += 1
        self.after(1, self.step)