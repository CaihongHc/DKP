import math

import numpy as np
import torch
from torch.nn import functional as F

from config import CONTROL_DT, DEPTH_HEIGHT, DEPTH_MAX_METERS, DEPTH_WIDTH


def preprocess_depth(depth_m):
    depth = torch.as_tensor(depth_m, dtype=torch.float32).detach()
    if depth.ndim != 2 or depth.numel() == 0:
        raise ValueError("depth_m must be a non-empty 2D depth image in metres")
    depth = torch.where(torch.isfinite(depth), depth, DEPTH_MAX_METERS).clamp(0.0, DEPTH_MAX_METERS)
    depth = depth.unsqueeze(0).unsqueeze(0)
    if depth.shape[-2:] != (DEPTH_HEIGHT, DEPTH_WIDTH):
        depth = F.interpolate(depth, size=(DEPTH_HEIGHT, DEPTH_WIDTH), mode="area")
    return (DEPTH_MAX_METERS - depth) / DEPTH_MAX_METERS


def build_state(
    goal_delta,
    yaw,
    velocity=(0.0, 0.0),
    *,
    velocity_is_body=False,
    dt=CONTROL_DT,
    horizon_steps=180,
    margin=0.3,
    max_speed=1.0,
):
    goal_delta = np.asarray(goal_delta, dtype=np.float32)
    velocity = np.asarray(velocity, dtype=np.float32)
    if goal_delta.shape != (2,) or velocity.shape != (2,):
        raise ValueError("goal_delta and velocity must each contain two values")
    scalars = np.asarray([yaw, dt, horizon_steps, margin, max_speed], dtype=float)
    if not (np.isfinite(scalars).all() and np.isfinite(goal_delta).all()
            and np.isfinite(velocity).all()):
        raise ValueError("state inputs must be finite")
    if dt <= 0 or horizon_steps <= 0 or max_speed <= 0 or margin < 0:
        raise ValueError("dt, horizon_steps, max_speed must be positive; margin >= 0")

    c, s = math.cos(yaw), math.sin(yaw)
    world_to_body = np.asarray([[c, s], [-s, c]], dtype=np.float32)
    target = np.clip((world_to_body @ goal_delta) / (dt * horizon_steps), -2.0, 2.0)
    local_velocity = velocity if velocity_is_body else world_to_body @ velocity
    features = [local_velocity, target, np.asarray([margin, max_speed], dtype=np.float32)]
    return np.concatenate(features).astype(np.float32)
