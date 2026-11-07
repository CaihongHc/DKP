import math

import torch

from config import LEARNING_RATE, MAX_GRAD_NORM, MIN_LR_RATIO, TRAINING_ITERATIONS, WEIGHT_DECAY


def build_optimizer(model, lr=LEARNING_RATE, num_iter=TRAINING_ITERATIONS,
                    weight_decay=WEIGHT_DECAY, min_lr_ratio=MIN_LR_RATIO):
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    if isinstance(num_iter, bool) or not isinstance(num_iter, int) or num_iter <= 0:
        raise ValueError("num_iter must be a positive integer")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    if not math.isfinite(min_lr_ratio) or not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be finite and in [0, 1]")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_iter, eta_min=lr * min_lr_ratio,
    )
    return optimizer, scheduler


def optimization_step(loss, model, optimizer, scheduler=None, max_grad_norm=MAX_GRAD_NORM):
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return grad_norm.detach()
