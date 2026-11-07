import torch
from torch import nn

from config import ACTION_DIM, DEPTH_HEIGHT, DEPTH_WIDTH, HIDDEN_DIM, STATE_DIM


class Model(nn.Module):

    def __init__(self, dim_obs=STATE_DIM, dim_action=ACTION_DIM):
        super().__init__()
        if dim_obs != STATE_DIM or dim_action != ACTION_DIM:
            raise ValueError("the paper model requires 6 states and 2 drive commands")
        self.dim_obs = dim_obs
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.05),
            nn.Flatten(),
            nn.Linear(128 * 3 * 4, HIDDEN_DIM, bias=False),
        )
        self.v_proj = nn.Linear(dim_obs, HIDDEN_DIM)
        self.v_proj.weight.data.mul_(0.5)
        self.fuse = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM, bias=False),
            nn.LeakyReLU(0.05),
        )
        self.ln = nn.LayerNorm(HIDDEN_DIM)
        self.gru = nn.GRUCell(HIDDEN_DIM, HIDDEN_DIM)
        self.recurrent_ln = nn.LayerNorm(HIDDEN_DIM)
        self.fc = nn.Linear(HIDDEN_DIM, dim_action, bias=False)
        self.fc.weight.data.mul_(0.1)
        self.act = nn.LeakyReLU(0.05)
        self._hx = None

    def reset(self):
        self._hx = None

    def forward(self, x: torch.Tensor, v: torch.Tensor, hx=None):
        if x.ndim != 4 or x.shape[1:] != (1, DEPTH_HEIGHT, DEPTH_WIDTH):
            raise ValueError("depth input must have shape (B, 1, 24, 32)")
        if v.shape != (x.shape[0], self.dim_obs):
            raise ValueError(f"state input must have shape (B, {self.dim_obs})")
        max_speed = v[:, -1:]
        if not torch.isfinite(max_speed).all() or (max_speed <= 0).any():
            raise ValueError("state max_speed must be finite and positive")
        img_feat = self.stem(x)
        v_feat = self.v_proj(v)
        x = self.fuse(torch.cat([img_feat, v_feat], dim=-1))
        x = self.ln(x)
        if hx is None:
            hx = self._hx
        if hx is None:
            hx = x.new_zeros((x.shape[0], self.gru.hidden_size))
        hx = self.gru(x, self.recurrent_ln(hx))
        self._hx = hx
        action = torch.tanh(self.fc(self.act(hx))) * max_speed
        return action, None, hx

    def forward_commands(self, x: torch.Tensor, v: torch.Tensor, hx=None):
        commands, _, hx = self.forward(x, v, hx)
        return commands, hx
