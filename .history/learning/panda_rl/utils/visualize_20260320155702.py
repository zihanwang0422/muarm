"""MuJoCo viewer overlay helpers for goals and trajectory lines."""
from __future__ import annotations

import mujoco
import numpy as np


def clear_overlay(viewer) -> None:
    viewer.user_scn.ngeom = 0


def draw_sphere(viewer, pos, radius=0.02, rgba=None) -> int:
    idx = viewer.user_scn.ngeom
    if idx >= viewer.user_scn.maxgeom:
        return idx
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[idx],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, 0.0, 0.0]),
        pos=np.array(pos, dtype=np.float64),
        mat=np.eye(3).ravel(),
        rgba=np.array(rgba or [1.0, 0.1, 0.1, 0.7], dtype=np.float32),
    )
    viewer.user_scn.ngeom = idx + 1
    return idx + 1


def draw_polyline(viewer, points, width=2.0, rgba=None) -> int:
    color = np.array(rgba or [0.1, 0.8, 1.0, 0.9], dtype=np.float32)
    if len(points) < 2:
        return viewer.user_scn.ngeom

    for i in range(len(points) - 1):
        idx = viewer.user_scn.ngeom
        if idx >= viewer.user_scn.maxgeom:
            break
        g = viewer.user_scn.geoms[idx]
        mujoco.mjv_connector(
            g,
            mujoco.mjtGeom.mjGEOM_LINE,
            width,
            np.array(points[i], dtype=np.float64),
            np.array(points[i + 1], dtype=np.float64),
        )
        g.rgba[:] = color
        viewer.user_scn.ngeom = idx + 1
    return viewer.user_scn.ngeom
