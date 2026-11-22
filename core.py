import numpy as np
from collections import namedtuple

RenderState = namedtuple("RenderState", ["buffer", "width", "height"])

# state = RenderState(
#     buffer=np.zeros((240, 320, 3), np.uint8),
#     width=320,
#     height=240,
# )