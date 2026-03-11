import os
import sys
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from alpha158_idx import bpda_idxmax, bpda_idxmin


def check_forward():
    x = torch.tensor([[2.0, 4.0, 1.0, 3.0]], requires_grad=True)
    imax = bpda_idxmax(x, tau=0.5, dim=-1)
    imin = bpda_idxmin(x, tau=0.5, dim=-1)
    print("[forward] idxmax (1-based):", imax.detach())
    print("[forward] idxmin (1-based):", imin.detach())


def check_grad():
    x = torch.tensor([[2.0, 4.0, 1.0, 3.0]], requires_grad=True)
    imax = bpda_idxmax(x, tau=0.5, dim=-1)
    loss = imax.sum()
    loss.backward()
    print("[grad] idxmax grad:", x.grad.detach())

    x2 = torch.tensor([[2.0, 4.0, 1.0, 3.0]], requires_grad=True)
    imin = bpda_idxmin(x2, tau=0.5, dim=-1)
    imin.sum().backward()
    print("[grad] idxmin grad:", x2.grad.detach())


if __name__ == "__main__":
    torch.manual_seed(0)
    check_forward()
    check_grad()
    print("module5 validation done")
