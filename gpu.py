import numpy as np
from colors import *
from core import RenderState
import gte
from numba import njit
    
def shaded_triangle(buffer, tri, colors):
    x0, y0 = tri[0, 0], tri[0, 1]
    x1, y1 = tri[1, 0], tri[1, 1]
    x2, y2 = tri[2, 0], tri[2, 1]

    area = gte.edge_function(x0, y0, x1, y1, x2, y2)
    if area <= 0:
        return
    
    h, w = buffer.shape[:2]
    
    x_list = tri[:, 0]
    y_list = tri[:, 1]

    x_min = max(x_list.min(), 0)
    x_max = min(x_list.max(), w - 1)
    y_min = max(y_list.min(), 0)
    y_max = min(y_list.max(), h - 1)

    if x_min > x_max or y_min > y_max:
        return
    
    xs_box = np.arange(x_min, x_max + 1)            # one entry per column of the box
    ys_box = np.arange(y_min, y_max + 1)[:, None]   # one entry per row of the box

    A1, B1, C1 = gte.edge_coeffs(x0, y0, x1, y1)
    A2, B2, C2 = gte.edge_coeffs(x1, y1, x2, y2)
    A3, B3, C3 = gte.edge_coeffs(x2, y2, x0, y0)

    e1 = A1 * xs_box + (B1 * ys_box + C1)
    e2 = A2 * xs_box + (B2 * ys_box + C2)
    e3 = A3 * xs_box + (B3 * ys_box + C3)

    inside = (e1 >= 0) & (e2 >= 0) & (e3 >= 0)

    ys, xs = np.nonzero(inside)
    xs_s = xs + x_min
    ys_s = ys + y_min

    a, b, c = gte.plane_coeffs(tri, colors)

    values = a * xs_s[:, None] + b * ys_s[:, None] + c
    
    for i in range(xs_s.shape[0]):
        x = xs_s[i]
        y = ys_s[i]
        for ch in range(3):
            buffer[y, x, ch] = np.uint8(values)

@njit(cache=True)
def textured_triangle(buffer, tri, uv, img):
    x0, y0 = tri[0, 0], tri[0, 1]
    x1, y1 = tri[1, 0], tri[1, 1]
    x2, y2 = tri[2, 0], tri[2, 1]

    area = gte.edge_function(x0, y0, x1, y1, x2, y2)
    if area <= 0:
        return
    
    h, w = buffer.shape[:2]
    
    x_list = tri[:, 0]
    y_list = tri[:, 1]

    x_min = max(x_list.min(), 0)
    x_max = min(x_list.max(), w - 1)
    y_min = max(y_list.min(), 0)
    y_max = min(y_list.max(), h - 1)

    if x_min > x_max or y_min > y_max:
        return
    
    xs_box = np.arange(x_min, x_max + 1)            # one entry per column of the box
    ys_box = np.arange(y_min, y_max + 1)[:, None]   # one entry per row of the box

    A1, B1, C1 = gte.edge_coeffs(x0, y0, x1, y1)
    A2, B2, C2 = gte.edge_coeffs(x1, y1, x2, y2)
    A3, B3, C3 = gte.edge_coeffs(x2, y2, x0, y0)

    e1 = A1 * xs_box + (B1 * ys_box + C1)
    e2 = A2 * xs_box + (B2 * ys_box + C2)
    e3 = A3 * xs_box + (B3 * ys_box + C3)

    inside = (e1 >= 0) & (e2 >= 0) & (e3 >= 0)

    ys, xs = np.nonzero(inside)
    xs_s = xs + x_min
    ys_s = ys + y_min

    size = np.array(img.shape[:2][::-1]) - 1     # (W-1, H-1), the largest u and v in texels
    uv_texel = uv * size   # TODO: scale the (3, 2) vertex UVs by size, once per triangle

    a, b, c = gte.plane_coeffs(tri, uv_texel)   # each has shape (2,): one for u, one for v

    t = (a * xs_s[:, None] + b * ys_s[:, None] + c).astype(np.uint64)
    ty, tx = t[:, 1], t[:, 0]

    for i in range(xs_s.shape[0]):
        x = xs_s[i]
        y = ys_s[i]
        tx = int(t[i, 0])
        ty = int(t[i, 1])
        for ch in range(3):
            buffer[y, x, ch] = np.uint8(img[ty, tx, ch])

    return

@njit(cache=True)
def shaded_textured_triangle(buffer, tri, uv, img, colors):
    x0, y0 = tri[0, 0], tri[0, 1]
    x1, y1 = tri[1, 0], tri[1, 1]
    x2, y2 = tri[2, 0], tri[2, 1]

    area = gte.edge_function(x0, y0, x1, y1, x2, y2)
    if area <= 0:
        return
    
    h, w = buffer.shape[:2]
    
    x_list = tri[:, 0]
    y_list = tri[:, 1]

    x_min = max(x_list.min(), 0)
    x_max = min(x_list.max(), w - 1)
    y_min = max(y_list.min(), 0)
    y_max = min(y_list.max(), h - 1)

    if x_min > x_max or y_min > y_max:
        return
    
    xs_box = np.arange(x_min, x_max + 1)            # one entry per column of the box
    ys_box = np.arange(y_min, y_max + 1)[:, None]   # one entry per row of the box

    A1, B1, C1 = gte.edge_coeffs(x0, y0, x1, y1)
    A2, B2, C2 = gte.edge_coeffs(x1, y1, x2, y2)
    A3, B3, C3 = gte.edge_coeffs(x2, y2, x0, y0)

    e1 = A1 * xs_box + (B1 * ys_box + C1)
    e2 = A2 * xs_box + (B2 * ys_box + C2)
    e3 = A3 * xs_box + (B3 * ys_box + C3)

    inside = (e1 >= 0) & (e2 >= 0) & (e3 >= 0)

    ys, xs = np.nonzero(inside)
    xs_s = xs + x_min
    ys_s = ys + y_min

    size = np.array((img.shape[1] - 1, img.shape[0] - 1))     # (W-1, H-1), the largest u and v in texels
    uv_texel = uv * size   # TODO: scale the (3, 2) vertex UVs by size, once per triangle

    a, b, c = gte.plane_coeffs(tri, np.hstack((colors, uv_texel)))   # each has shape (2,): one for u, one for v

    ca, cb, cc = a[:3], b[:3], c[:3]
    ua, ub, uc = a[3:], b[3:], c[3:]

    t = (ua * xs_s[:, None] + ub * ys_s[:, None] + uc).astype(np.int64)
    ty, tx = t[:, 1], t[:, 0]

    gouraud = ca * xs_s[:, None] + cb * ys_s[:, None] + cc

    for i in range(xs_s.shape[0]):
        x = xs_s[i]
        y = ys_s[i]
        tx = int(t[i, 0])
        ty = int(t[i, 1])
        for ch in range(3):
            buffer[y, x, ch] = np.uint8(img[ty, tx, ch] * gouraud[i, ch] / 255)

    return