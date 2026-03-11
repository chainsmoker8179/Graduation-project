import os
import sys
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from alpha158_torch import TorchFactorExtractor


def main():
    B, L = 2, 70
    x = (torch.rand(B, L, 6) * 10.0 + 1.0).detach().requires_grad_()
    extractor = TorchFactorExtractor()
    feats = extractor(x)
    print("[shape] feats:", feats.shape)

    # gradient check
    loss = feats.mean()
    loss.backward()
    print("[grad] x.grad mean:", x.grad.abs().mean().item())


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
