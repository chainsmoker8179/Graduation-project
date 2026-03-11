import os
import sys
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from alpha158_softsort import soft_sort, soft_rank


def check_soft_sort():
    x = torch.tensor([[2.0, 4.0, 1.0, 3.0]], requires_grad=True)
    s = soft_sort(x, tau=0.1, direction="DESCENDING")
    print("[soft_sort] input:", x.detach())
    print("[soft_sort] output:", s.detach())
    # hard sort for reference
    hard = torch.sort(x, descending=True, dim=-1).values
    print("[soft_sort] hard:", hard.detach())


def check_soft_rank():
    x = torch.tensor([[2.0, 4.0, 1.0, 3.0]], requires_grad=True)
    r = soft_rank(x, tau=0.1, direction="DESCENDING", pct=True)
    print("[soft_rank] pct:", r.detach())
    # hard rank pct for reference
    hard_order = torch.argsort(x, descending=True, dim=-1)
    # build hard rank 1..N
    B, N = x.shape
    hard_rank = torch.empty_like(x)
    for b in range(B):
        for idx, pos in enumerate(hard_order[b]):
            hard_rank[b, pos] = idx + 1
    hard_rank_pct = (hard_rank - 1) / (N - 1)
    print("[soft_rank] hard pct:", hard_rank_pct)


def check_grad():
    x = torch.tensor([[2.0, 4.0, 1.0, 3.0]], requires_grad=True)
    s = soft_sort(x, tau=0.5, direction="DESCENDING")
    loss = s.sum()
    loss.backward()
    print("[grad] soft_sort grad:", x.grad.detach())

    x2 = torch.tensor([[2.0, 4.0, 1.0, 3.0]], requires_grad=True)
    r = soft_rank(x2, tau=0.5, direction="DESCENDING", pct=True)
    r.sum().backward()
    print("[grad] soft_rank grad:", x2.grad.detach())


if __name__ == "__main__":
    torch.manual_seed(0)
    check_soft_sort()
    check_soft_rank()
    check_grad()
    print("module3 validation done")
