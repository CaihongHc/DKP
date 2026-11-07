from collections import deque

import torch

from config import ACTION_DELAY_STEPS


class ActionDelayBuffer:

    def __init__(self, steps=ACTION_DELAY_STEPS):
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        self.steps = steps
        self.reset()

    def reset(self):
        self._queue = deque()

    def push(self, command):
        if command.ndim < 1 or command.shape[-1] != 2 or not command.is_floating_point():
            raise ValueError("command must be a floating-point tensor of shape (..., 2)")
        if self.steps == 0:
            return command.clone()
        if not self._queue:
            self._queue.extend(torch.zeros_like(command) for _ in range(self.steps))
        reference = self._queue[0]
        if (command.shape != reference.shape or command.device != reference.device
                or command.dtype != reference.dtype):
            raise ValueError("call reset() before changing command shape, device, or dtype")
        self._queue.append(command.clone())
        return self._queue.popleft()
