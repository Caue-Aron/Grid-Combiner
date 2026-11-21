from gpu import GPU
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import time

class VOut(tk.Tk):
    def __init__(self, gpu:GPU, title='Demo'):
        super().__init__()
        self._gpu = gpu
        self._gte = self._gpu._gte
        self._core = self._gpu._core

        self.title(title)
        self.scale = 3
        self.geometry(f'{gpu._core._scr_width*self.scale}x{gpu._core._scr_height*self.scale}')

        self.fps, self.total = 30, 60
        self.frame_i = 0
        self.after(0, self.step)
        self.last = 0
        self.last = time.perf_counter()
        self.label = tk.Label(self)
        self.label.pack()

    def render(self, t, dt):
        # your software renderer goes here; return an (H, W, 3) uint8 array
        return np.empty((240, 320, 3), dtype=np.uint8)

    def step(self):
        now = time.perf_counter()
        dt = now - self.last
        self.last = now
        
        t = self.frame_i / self.fps
        t0 = time.perf_counter()
        frame = self.render(t, dt)
        ms = (time.perf_counter() - t0) * 1000

        print(ms)

        img = Image.fromarray(frame).resize((320*self.scale, 240*self.scale), Image.NEAREST)
        self.photo = ImageTk.PhotoImage(img)   # keep a reference on self
        self.label.config(image=self.photo)

        self.frame_i += 1
        self.after(1, self.step)