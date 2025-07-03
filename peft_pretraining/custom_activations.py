"""Activation functions used in GPAS."""
import torch
import torch.nn.functional as F
from transformers import activations


def scale_silu(x, beta=8):
    return F.sigmoid(x * beta) * x


ACT2FN = {
    'identity': lambda x: x,
    'silu': F.silu,
    'relu': F.relu,
    'tanh': F.tanh,
    'leaky_relu': F.leaky_relu,
    'scale_silu': scale_silu,
}
ACT2FN.update({k: v for k, v in activations.ACT2FN.items() if k not in ACT2FN})


if __name__ == "__main__":
    print(ACT2FN.keys())