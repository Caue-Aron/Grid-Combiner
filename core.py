import numpy as np

class Core:
    _pixel_resolution = np.uint8

    def __init__(self, scr_width=320, scr_height=240):
        self._scr_width = scr_width
        self._scr_height = scr_height
        
        self._buffer = np.empty(
            (self._scr_height, self._scr_width, 3),
            dtype=self._pixel_resolution
        )
        
        self._grid_yx = np.mgrid[:self._scr_height, :self._scr_width]
        self._grid_y, self._grid_x = self._grid_yx