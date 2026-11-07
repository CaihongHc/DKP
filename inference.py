import numpy as np
import torch

from config import ACTION_DIM, STATE_DIM
from model import Model
from preprocessing import preprocess_depth


class DKPNetPolicy:

    def __init__(self, checkpoint, device="cpu", wheel_base=0.5):
        self.device = torch.device(device)
        self.wheel_base = wheel_base
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.dim_obs = STATE_DIM
        if state_dict["v_proj.weight"].shape[1] != STATE_DIM or state_dict["fc.weight"].shape[0] != ACTION_DIM:
            raise ValueError("the paper checkpoint requires 6 states and 2 drive commands")
        self.model = Model()
        if not {"recurrent_ln.weight", "recurrent_ln.bias"}.issubset(state_dict):
            raise ValueError(
                "checkpoint lacks recurrent_ln parameters required by the paper-aligned model; "
                "use a checkpoint trained with recurrent hidden-state normalization"
            )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device).eval()
        self.reset()

    def reset(self):
        self.model.reset()

    @torch.inference_mode()
    def predict(self, depth_m, state):
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self.dim_obs,) or not np.isfinite(state).all():
            raise ValueError(f"state must contain {self.dim_obs} finite values")
        if state[-1] <= 0 or state[-2] < 0:
            raise ValueError("state max_speed must be positive and margin >= 0")
        depth = preprocess_depth(depth_m).to(self.device)
        state_t = torch.as_tensor(state, device=self.device).unsqueeze(0)
        commands, _, _ = self.model(depth, state_t)
        wheel_speeds = commands[0]
        return {
            "wheel_speeds": wheel_speeds.cpu().numpy(),
            "linear_velocity": None,
            "angular_velocity": None,
        }
