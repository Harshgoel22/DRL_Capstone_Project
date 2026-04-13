import numpy as np
import torch
import torch.nn as nn
import os

# updated best model weights mentioned in report (giving more success)

# Globals
_MODEL = None
_HX, _CX = None, None

LSTM_H_DIM = 128
DECAY = 0.95          # soft reset factor
MAX_STEPS = 1000      # fallback hard reset

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
    wpath = os.path.join(submission_dir, "./model_weights/best_weights_on_success_basis.pth")

    class KnowledgeNet(nn.Module):
        def __init__(self, inDim=18, outDim=5, f_hDim=[512, 256], lstm_hDim=128,
                     for_hDim=[256, 256], inv_hDim=[256, 256], activation=nn.ReLU):
            super(KnowledgeNet, self).__init__()
            self.activation = activation

            # Encoder
            prevDim = inDim
            layers = []
            for nextDim in f_hDim:
                layers.append(nn.Linear(prevDim, nextDim))
                layers.append(self.activation())
                prevDim = nextDim
            self.features = nn.Sequential(*layers)

            # LSTM
            self.lstmCell = nn.LSTMCell(prevDim, lstm_hDim)

            # Q head
            self.values = nn.Linear(lstm_hDim, 1)
            self.advantages = nn.Linear(lstm_hDim, outDim)

            # bad state head
            self.bad_state_head = nn.Linear(lstm_hDim, 1)

            # Forward model
            prevDim = lstm_hDim + outDim
            forward_layers = []
            for nextDim in for_hDim:
                forward_layers.append(nn.Linear(prevDim, nextDim))
                forward_layers.append(self.activation())
                prevDim = nextDim
            forward_layers.append(nn.Linear(prevDim, lstm_hDim))
            self.forward_model = nn.Sequential(*forward_layers)

            # Inverse model
            prevDim = 2 * lstm_hDim
            inverse_layers = []
            for nextDim in inv_hDim:
                inverse_layers.append(nn.Linear(prevDim, nextDim))
                inverse_layers.append(self.activation())
                prevDim = nextDim
            inverse_layers.append(nn.Linear(prevDim, outDim))
            self.inverse_model = nn.Sequential(*inverse_layers)

            # latent represenation
            self.latent_proj = nn.Sequential(
                nn.Linear(lstm_hDim, lstm_hDim),
                nn.ReLU(),
                nn.LayerNorm(lstm_hDim)
            )

            # Bump Head (for decinding which bump is bad)
            self.bump_head = nn.Sequential(
                nn.Linear(lstm_hDim+2, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

        def forward(self, x, hx, cx):
            x_f = self.features(x)
            hx, cx = self.lstmCell(x_f, (hx, cx))

            z_t = self.latent_proj(hx)

            v = self.values(z_t)
            adv = self.advantages(z_t)
            q = v + (adv - adv.mean(dim=1, keepdim=True))
            
            return q, hx, cx

    # Init Model
    _MODEL = KnowledgeNet(
        f_hDim=[324, 256],
        lstm_hDim=LSTM_H_DIM,
        for_hDim=[224, 128],
        inv_hDim=[224, 128]
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
        q, _HX, _CX = _MODEL(x, _HX, _CX)

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