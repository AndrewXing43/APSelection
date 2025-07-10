import torch
import torch.nn as nn
import numpy as np
import random

# ------------------------
# Loss Functions
# ------------------------
def get_loss_fn(name="smoothl1"):
    if name == "mse":
        return nn.MSELoss()
    elif name == "smoothl1":
        return nn.SmoothL1Loss()
    else:
        raise ValueError("Unsupported loss function: {}".format(name))

# ------------------------
# RMSE Metric
# ------------------------
def compute_rmse(preds, targets):
    """
    preds: (N, 2)
    targets: (N, 2)
    """
    return torch.sqrt(torch.mean((preds - targets) ** 2)).item()

# ------------------------
# Save/Load Model
# ------------------------
def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model

# ------------------------
# Random Seed
# ------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
