import sys

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from utils.dilate_loss import dilate_loss
# from utils.dilate_loss_cache import dilate_loss


def dpp_loss(predictions, target, alpha, gamma, device):
    recon_loss, _, _ = dilate_loss(predictions, target, alpha, gamma, device)

    nsamples = 1
    eps = torch.finfo(torch.float32).eps

    dilate, dtw, tdi = dilate_loss(predictions, predictions, alpha, gamma, device)
    Lambda = 1 / (torch.abs(dtw) + eps)
    K = torch.exp(-Lambda * dtw)
    I = torch.eye((nsamples)).to(device)
    M = I - torch.inverse(K+I)
    diversity_loss_shape = -torch.trace(M)

    Lambda = 1 / (torch.abs(tdi) + eps)
    K = torch.exp(10 * Lambda * tdi)
    I = torch.eye((nsamples)).to(device)
    M = I - torch.inverse(K+I)
    diversity_loss_time = -torch.trace(M)

    return recon_loss + diversity_loss_shape + diversity_loss_time
