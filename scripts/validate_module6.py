import os
import sys
import torch
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from alpha158_regression import rolling_slope, rolling_rsquare, rolling_resi, rolling_corr


def numpy_slope(x):
    t = np.arange(1, len(x) + 1)
    t_mean = t.mean()
    x_mean = x.mean()
    cov = np.sum((t - t_mean) * (x - x_mean))
    var_t = np.sum((t - t_mean) ** 2)
    return cov / var_t


def numpy_rsquare(x):
    t = np.arange(1, len(x) + 1)
    t_mean = t.mean()
    x_mean = x.mean()
    cov = np.sum((t - t_mean) * (x - x_mean))
    var_t = np.sum((t - t_mean) ** 2)
    var_x = np.sum((x - x_mean) ** 2)
    return (cov ** 2) / (var_t * var_x + 1e-12)


def numpy_resi(x):
    t = np.arange(1, len(x) + 1)
    t_mean = t.mean()
    x_mean = x.mean()
    cov = np.sum((t - t_mean) * (x - x_mean))
    var_t = np.sum((t - t_mean) ** 2)
    slope = cov / var_t
    intercept = x_mean - slope * t_mean
    y_hat = intercept + slope * t[-1]
    return x[-1] - y_hat


def numpy_corr(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    x_mean = x.mean()
    y_mean = y.mean()
    cov = np.sum((x - x_mean) * (y - y_mean))
    var_x = np.sum((x - x_mean) ** 2)
    var_y = np.sum((y - y_mean) ** 2)
    return cov / (np.sqrt(var_x) * np.sqrt(var_y) + 1e-12)


def check_single_window():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    y = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
    w = 5

    slope = rolling_slope(x, w, dim=1).item()
    rsq = rolling_rsquare(x, w, dim=1).item()
    resi = rolling_resi(x, w, dim=1).item()
    corr = rolling_corr(x, y, w, dim=1).item()

    xn = x.numpy().squeeze(0)
    yn = y.numpy().squeeze(0)

    print("[slope] torch:", slope, "numpy:", numpy_slope(xn))
    print("[rsq]   torch:", rsq, "numpy:", numpy_rsquare(xn))
    print("[resi]  torch:", resi, "numpy:", numpy_resi(xn))
    print("[corr]  torch:", corr, "numpy:", numpy_corr(xn, yn))


def check_grad():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], requires_grad=True)
    y = rolling_slope(x, 5, dim=1).sum() + rolling_rsquare(x, 5, dim=1).sum() + rolling_resi(x, 5, dim=1).sum()
    y.backward()
    print("[grad] slope/rsq/resi grad:", x.grad)

    x2 = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], requires_grad=True)
    y2 = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]], requires_grad=True)
    c = rolling_corr(x2, y2, 5, dim=1).sum()
    c.backward()
    print("[grad] corr grad x:", x2.grad)
    print("[grad] corr grad y:", y2.grad)

    # Non-perfect correlation example
    x3 = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], requires_grad=True)
    y3 = torch.tensor([[5.0, 4.2, 3.1, 2.2, 1.0]], requires_grad=True)
    c2 = rolling_corr(x3, y3, 5, dim=1).sum()
    c2.backward()
    print("[grad] corr (non-perfect) grad x:", x3.grad)
    print("[grad] corr (non-perfect) grad y:", y3.grad)


if __name__ == "__main__":
    check_single_window()
    check_grad()
    print("module6 validation done")
