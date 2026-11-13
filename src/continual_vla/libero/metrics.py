"""Continual-learning metrics on the success matrix of a task sequence (LIBERO; Liu et al., 2023, and the VLA papers
that report it, e.g. CLARE, Romer et al., 2026).

r[m][n] is the success rate on task n after learning the first m + 1 tasks of a sequence of N (n <= m):

  FWT   mean over tasks of r[n][n], each task's success right after it is learned
  NBT   mean over tasks n < N - 1 of the mean drop r[n][n] - r[m][n] over the later stages m > n (forgetting;
        lower is better, negative when a task improves later)
  AUC   mean over tasks of the mean of r[m][n] over the stages m >= n, from learning the task onwards
  final mean of r[N - 1][n], every task after the whole sequence

LIBERO's original FWT is instead the mean success over training checkpoints of each task, and its r[n][n] is the best
checkpoint's; here every stage is scored once, at its end.
"""

import numpy as np


def sequence_metrics(r):
    """`r`: N x N (lower triangle used; rows with missing stages are allowed, as NaN). Values as in r."""
    r = np.asarray(r, dtype=float)
    n_tasks = r.shape[0]
    done = int(np.sum(~np.isnan(np.diag(r))))
    fwt = float(np.nanmean(np.diag(r)[:done]))
    drops = [np.mean([r[n, n] - r[m, n] for m in range(n + 1, done)]) for n in range(done - 1)]
    auc = [np.mean([r[m, n] for m in range(n, done)]) for n in range(done)]
    out = {"tasks_learned": done, "FWT": fwt, "NBT": float(np.mean(drops)) if drops else float("nan"),
           "AUC": float(np.mean(auc)), "final": float(np.mean(r[done - 1, :done]))}
    out["complete"] = done == n_tasks
    return out
