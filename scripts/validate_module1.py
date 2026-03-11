import os
import sys
import torch

# Ensure repo root is on PYTHONPATH
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from alpha158_diff_ops import (
    bpda_max_pair,
    bpda_min_pair,
    smooth_max_pair,
    smooth_min_pair,
    bpda_greater,
    bpda_less,
    DEFAULT_TEMP_CMP,
)


def check_forward():
    a = torch.tensor([2.0, 1.0, -1.0, 3.0], requires_grad=True)
    b = torch.tensor([1.0, 2.0, -2.0, 3.0], requires_grad=True)

    out_max = bpda_max_pair(a, b, tau=5.0)
    out_min = bpda_min_pair(a, b, tau=5.0)

    hard_max = torch.maximum(a, b)
    hard_min = torch.minimum(a, b)

    print("[forward] bpda_max == hard max:", torch.allclose(out_max, hard_max))
    print("[forward] bpda_min == hard min:", torch.allclose(out_min, hard_min))

    # Greater/Less hard forward equivalence
    g = bpda_greater(a, b, temperature=DEFAULT_TEMP_CMP)
    l = bpda_less(a, b, temperature=DEFAULT_TEMP_CMP)
    print("[forward] bpda_greater == (a>b):", torch.equal(g, (a > b).to(dtype=a.dtype)))
    print("[forward] bpda_less == (a<b):", torch.equal(l, (a < b).to(dtype=a.dtype)))


def check_gradients():
    a = torch.tensor([2.0, 1.0, -1.0, 3.0], requires_grad=True)
    b = torch.tensor([1.0, 2.0, -2.0, 3.0], requires_grad=True)

    out = bpda_max_pair(a, b, tau=5.0) + bpda_min_pair(a, b, tau=5.0)
    loss = out.sum()
    loss.backward(retain_graph=True)

    print("[grad] max/min a.grad:", a.grad.detach())
    print("[grad] max/min b.grad:", b.grad.detach())
    print("[grad] nonzero a.grad:", int((a.grad.abs() > 1e-8).sum().item()))
    print("[grad] nonzero b.grad:", int((b.grad.abs() > 1e-8).sum().item()))

    # reset grads
    a.grad = None
    b.grad = None

    g = bpda_greater(a, b, temperature=DEFAULT_TEMP_CMP)
    l = bpda_less(a, b, temperature=DEFAULT_TEMP_CMP)
    (g.sum() + l.sum()).backward()

    print("[grad] greater+less a.grad:", a.grad.detach())
    print("[grad] greater+less b.grad:", b.grad.detach())
    print("[grad] nonzero a.grad:", int((a.grad.abs() > 1e-8).sum().item()))
    print("[grad] nonzero b.grad:", int((b.grad.abs() > 1e-8).sum().item()))

    # Show gradients without cancellation (greater only / less only)
    ag = torch.tensor([2.0, 1.0], requires_grad=True)
    bg = torch.tensor([1.0, 2.0], requires_grad=True)
    bpda_greater(ag, bg, temperature=DEFAULT_TEMP_CMP).sum().backward()
    print("[grad] greater-only a.grad:", ag.grad.detach())
    print("[grad] greater-only b.grad:", bg.grad.detach())

    al = torch.tensor([2.0, 1.0], requires_grad=True)
    bl = torch.tensor([1.0, 2.0], requires_grad=True)
    bpda_less(al, bl, temperature=DEFAULT_TEMP_CMP).sum().backward()
    print("[grad] less-only a.grad:", al.grad.detach())
    print("[grad] less-only b.grad:", bl.grad.detach())


def check_temperature_sensitivity():
    a = torch.tensor([2.0, 1.0, -1.0, 3.0], requires_grad=True)
    b = torch.tensor([1.0, 2.0, -2.0, 3.0], requires_grad=True)

    for tau in [1.0, 5.0, 10.0]:
        a.grad = None
        b.grad = None
        out = smooth_max_pair(a, b, tau=tau).sum() + smooth_min_pair(a, b, tau=tau).sum()
        out.backward()
        grad_norm = a.grad.norm().item() + b.grad.norm().item()
        print(f"[tau] tau={tau} grad_norm={grad_norm:.6f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    check_forward()
    check_gradients()
    check_temperature_sensitivity()
