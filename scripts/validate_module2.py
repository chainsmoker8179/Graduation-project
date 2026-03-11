import os
import sys
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from alpha158_rolling import (
    rolling_unfold,
    rolling_sum,
    rolling_mean,
    rolling_std,
    rolling_max,
    rolling_min,
    right_align,
)


def check_unfold_shape():
    x = torch.arange(1, 6, dtype=torch.float32).view(1, 5)  # (1,5)
    xu = rolling_unfold(x, window=3, dim=1)
    print("[unfold] shape:", tuple(xu.shape), "expected", (1, 3, 3))
    print("[unfold] values:", xu.squeeze(0))


def check_basic_reductions():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])  # (1,5)
    s = rolling_sum(x, window=3, dim=1)
    m = rolling_mean(x, window=3, dim=1)
    sd = rolling_std(x, window=3, dim=1, unbiased=False)
    mx = rolling_max(x, window=3, dim=1)
    mn = rolling_min(x, window=3, dim=1)

    print("[sum]", s)
    print("[mean]", m)
    print("[std]", sd)
    print("[max]", mx)
    print("[min]", mn)

    # expected sums: [1+2+3, 2+3+4, 3+4+5] = [6,9,12]
    assert torch.allclose(s, torch.tensor([[6.0, 9.0, 12.0]]))


def check_right_align():
    a = torch.arange(10.0).view(1, 10)
    b = torch.arange(6.0).view(1, 6)
    ar, br = right_align(a, b, dim=1)
    print("[align] shapes:", tuple(ar.shape), tuple(br.shape))
    print("[align] ar last 6:", ar)
    print("[align] br:", br)
    assert ar.shape[1] == br.shape[1] == 6
    assert torch.allclose(ar, torch.arange(4.0, 10.0).view(1, 6))


if __name__ == "__main__":
    torch.manual_seed(0)
    check_unfold_shape()
    check_basic_reductions()
    check_right_align()
    print("module2 validation done")
