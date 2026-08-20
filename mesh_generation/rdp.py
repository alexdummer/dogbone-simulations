"""Ramer-Douglas-Peucker polyline simplification."""

import numpy as np


def rdp(points, epsilon):
    """Simplify an open polyline while preserving its shape within epsilon."""
    if len(points) < 3:
        return points
    start, end = points[0], points[-1]
    line = end - start
    line_len = np.hypot(*line)
    if line_len == 0:
        d = np.hypot(*(points[1:-1] - start).T)
    else:
        vec = points[1:-1] - start
        cross2d = line[0] * vec[:, 1] - line[1] * vec[:, 0]
        d = np.abs(cross2d) / line_len
    if len(d) == 0:
        return np.array([start, end])
    idx = np.argmax(d)
    dmax = d[idx]
    if dmax > epsilon:
        left = rdp(points[: idx + 2], epsilon)
        right = rdp(points[idx + 1 :], epsilon)
        return np.vstack([left[:-1], right])
    else:
        return np.array([start, end])
