import numpy as np
import torch
import torch.nn as nn
import os

# Globals
_MODEL = None
_HX, _CX = None, None

LSTM_H_DIM = 128
DECAY = 0.95          # soft reset factor
MAX_STEPS = 800       # fallback hard reset

step_counter = 0

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

def reset_hidden_state():
    global _HX, _CX
    _HX = torch.zeros(1, LSTM_H_DIM)
    _CX = torch.zeros(1, LSTM_H_DIM)


def _load_once():
    global _MODEL
    if _MODEL is not None:
        return

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "weights.pth")

    class KnowledgeNet(nn.Module):
        def __init__( self, state_dim=18, action_dim=5, feature_dims=[256, 256],
            lstm_dim=128, forward_dims=[256], inverse_dims=[256]
        ):
            super().__init__()

            # feature extractor
            self.state_dim = state_dim
            self.action_dim = action_dim

            layers = []
            prev = self.state_dim
            for dim in feature_dims:
                layers.append(nn.Linear(prev, dim))
                layers.append(nn.ReLU())
                prev = dim
            self.feature_net = nn.Sequential(*layers)

            # lstm
            self.lstm = nn.LSTMCell(prev, lstm_dim)

            # latent projection
            self.latent_proj = nn.Sequential(
                nn.Linear(lstm_dim, lstm_dim),
                nn.ReLU(),
                nn.LayerNorm(lstm_dim)
            )

            # Policy Head (Q-values)
            self.q_head = nn.Linear(lstm_dim, self.action_dim)

            # value head
            self.value_head = nn.Linear(lstm_dim, 1)

            # Forward Model
            f_layers = []
            prev = lstm_dim + self.action_dim
            for dim in forward_dims:
                f_layers.append(nn.Linear(prev, dim))
                f_layers.append(nn.ReLU())
                prev = dim
            f_layers.append(nn.Linear(prev, lstm_dim))
            self.forward_model = nn.Sequential(*f_layers)

            # inverse Model
            i_layers = []
            prev = 2 * lstm_dim
            for dim in inverse_dims:
                i_layers.append(nn.Linear(prev, dim))
                i_layers.append(nn.ReLU())
                prev = dim
            i_layers.append(nn.Linear(prev, self.action_dim))
            self.inverse_model = nn.Sequential(*i_layers)

        def forward(self, x, hx, cx):
            x = self.feature_net(x)

            hx, cx = self.lstm(x, (hx, cx))
            z = self.latent_proj(hx)
            q = self.q_head(z)

            return q, z, hx, cx

    # Init Model
    _MODEL = KnowledgeNet(
            state_dim=18,
            action_dim=5,
            feature_dims=[324, 256],
            lstm_dim=LSTM_H_DIM,
            forward_dims=[224, 128],
            inverse_dims=[224, 128]
        )

    # Load weights
    _MODEL.load_state_dict(torch.load(wpath, map_location="cpu"))
    _MODEL.eval()

    # Init hidden state
    reset_hidden_state()


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _HX, _CX, step_counter

    _load_once()

    # Convert observation
    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        q, _, _HX, _CX = _MODEL(x, _HX, _CX)

        _HX = (_HX * DECAY).detach()
        _CX = (_CX * DECAY).detach()

    # Action selection (greedy)
    action_idx = int(torch.argmax(q).item())

    # Step tracking
    step_counter += 1

    if step_counter >= MAX_STEPS:
        reset_hidden_state()
        step_counter = 0

    return ACTIONS[action_idx]