import os
import sys
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from alpha158_quantile import soft_quantile_window, bpda_quantile_window


def check_soft_quantile():
    x = torch.tensor([[1.0, 3.0, 2.0, 4.0]], requires_grad=True)
    for q in [0.2, 0.5, 0.8]:
        y = soft_quantile_window(x, q, reg_strength=0.5, pick_strength=0.3)
        hard = torch.quantile(x, q, dim=-1)
        print(f"[soft_quantile] q={q} soft={y.detach()} hard={hard.detach()}")


def check_bpda_forward():
    x = torch.tensor([[1.0, 3.0, 2.0, 4.0]], requires_grad=True)
    y = bpda_quantile_window(x, 0.5, reg_strength=0.5, pick_strength=0.3)
    hard = torch.quantile(x, 0.5, dim=-1)
    print("[bpda] forward matches hard:", torch.allclose(y, hard))


def check_grad():
    x = torch.tensor([[1.0, 3.0, 2.0, 4.0]], requires_grad=True)
    y = bpda_quantile_window(x, 0.5, reg_strength=0.5, pick_strength=0.3)
    y.sum().backward()
    print("[grad] bpda quantile grad:", x.grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    check_soft_quantile()
    check_bpda_forward()
    check_grad()
    print("module4 validation done")
