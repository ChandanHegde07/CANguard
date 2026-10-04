"""Representation-density scorers for Phase 4B.

All density statistics are fitted on **source normal** representations only.
No target sample influences the mean, covariance, regularization, feature
standardization, k, or threshold. This lets us compare anomaly scores on the
*same frozen representation* and separate representation quality from scoring.

Scores operate on contextual embeddings ``z`` of shape ``[N, d]``:

* ``mahalanobis``  — regularized full (or diagonal) Mahalanobis distance;
* ``knn``          — mean Euclidean distance to the k nearest source normals;
* ``centroid``     — Euclidean distance to the source-normal mean;
* ``probe``        — supervised linear probe (representation *diagnostic* only,
  never a zero-shot unsupervised detector).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Feature standardization (source-only, fixed transform)
# ---------------------------------------------------------------------------
def fit_standardizer(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = Z.mean(axis=0)
    sd = Z.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return mu.astype(np.float64), sd.astype(np.float64)


def apply_standardizer(Z: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (Z - mu) / sd


# ---------------------------------------------------------------------------
# Mahalanobis
# ---------------------------------------------------------------------------
def fit_mahalanobis(Z: np.ndarray, shrinkage: float = 0.1, diagonal: bool = False) -> dict:
    """Regularized normal statistics fitted on source normals.

    Full covariance uses shrinkage ``Sigma_reg = (1-s) Sigma + s * tr(Sigma)/d I``
    (a documented, standard regularizer); ``diagonal=True`` uses a diagonal
    covariance. ``shrinkage`` is fixed a priori, not tuned on the target.
    """
    Z = np.asarray(Z, dtype=np.float64)
    mu = Z.mean(axis=0)
    d = Z.shape[1]
    if diagonal:
        var = Z.var(axis=0)
        var = np.where(var < 1e-8, 1e-8, var)
        prec = 1.0 / var
        return {"mode": "diag", "mu": mu, "prec": prec}
    Xc = Z - mu
    cov = (Xc.T @ Xc) / max(1, len(Z) - 1)
    lam = float(shrinkage)
    cov_reg = (1 - lam) * cov + lam * (np.trace(cov) / d) * np.eye(d)
    prec = np.linalg.inv(cov_reg)
    return {"mode": "full", "mu": mu, "prec": prec}


def score_mahalanobis(Z: np.ndarray, stats: dict) -> np.ndarray:
    Xc = np.asarray(Z, dtype=np.float64) - stats["mu"]
    if stats["mode"] == "diag":
        return (Xc**2 * stats["prec"]).sum(axis=1)
    return np.einsum("ij,jk,ik->i", Xc, stats["prec"], Xc)


# ---------------------------------------------------------------------------
# Centroid
# ---------------------------------------------------------------------------
def fit_centroid(Z: np.ndarray) -> np.ndarray:
    return np.asarray(Z, dtype=np.float64).mean(axis=0)


def score_centroid(Z: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(Z, dtype=np.float64) - mu, axis=1)


# ---------------------------------------------------------------------------
# k-NN
# ---------------------------------------------------------------------------
def fit_knn_reference(Z: np.ndarray, max_refs: int = 20000, seed: int = 0) -> np.ndarray:
    Z = np.asarray(Z, dtype=np.float64)
    if len(Z) <= max_refs:
        return Z
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), size=max_refs, replace=False)
    return Z[idx]


def score_knn(
    Z: np.ndarray, refs: np.ndarray, k: int = 5, max_query: int = 40000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Mean distance to k nearest source normals.

    Returns ``(scores, query_index)``. If the query set exceeds ``max_query`` a
    seeded subset is used; the caller maps scores back via ``query_index``.
    """
    from sklearn.neighbors import NearestNeighbors

    Z = np.asarray(Z, dtype=np.float64)
    if len(Z) > max_query:
        rng = np.random.default_rng(seed)
        q = np.sort(rng.choice(len(Z), size=max_query, replace=False))
    else:
        q = np.arange(len(Z))
    k = int(min(k, len(refs)))
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(refs)
    dist, _ = nn.kneighbors(Z[q])
    return dist.mean(axis=1), q


# ---------------------------------------------------------------------------
# Representation-shift diagnostics
# ---------------------------------------------------------------------------
def _subsample(Z: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    if len(Z) <= n:
        return Z
    rng = np.random.default_rng(seed)
    return Z[rng.choice(len(Z), size=n, replace=False)]


def mmd_rbf(Zs: np.ndarray, Zt: np.ndarray, max_samples: int = 1500, seed: int = 0) -> float:
    """Biased MMD^2 with an RBF kernel (median-heuristic bandwidth).

    Pairwise distances use the Gram trick so memory is O(n^2), not O(n^2 d).
    """
    Zs = _subsample(np.asarray(Zs, dtype=np.float64), max_samples, seed)
    Zt = _subsample(np.asarray(Zt, dtype=np.float64), max_samples, seed + 1)
    if len(Zs) < 2 or len(Zt) < 2:
        return float("nan")
    both = np.vstack([Zs, Zt])
    med = float(np.median(np.sqrt(_sq(both, both))))
    gamma = 1.0 / (med**2 + 1e-12)
    kxx = np.exp(-gamma * _sq(Zs, Zs)).mean()
    kyy = np.exp(-gamma * _sq(Zt, Zt)).mean()
    kxy = np.exp(-gamma * _sq(Zs, Zt)).mean()
    return float(kxx + kyy - 2 * kxy)


def _sq(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return ((A**2).sum(axis=1)[:, None] + (B**2).sum(axis=1)[None, :] - 2 * A @ B.T).clip(min=0)


def representation_shift(Zs: np.ndarray, Zt: np.ndarray, seed: int = 0) -> dict:
    """Centroid distance, mean SMD, and MMD between source and target normals."""
    Zs = np.asarray(Zs, dtype=np.float64)
    Zt = np.asarray(Zt, dtype=np.float64)
    ms, mt = Zs.mean(axis=0), Zt.mean(axis=0)
    ss, st = Zs.std(axis=0), Zt.std(axis=0)
    pooled = np.sqrt((ss**2 + st**2) / 2) + 1e-12
    smd = (mt - ms) / pooled
    return {
        "centroid_distance": float(np.linalg.norm(mt - ms)),
        "mean_abs_smd": float(np.mean(np.abs(smd))),
        "median_abs_smd": float(np.median(np.abs(smd))),
        "max_abs_smd": float(np.max(np.abs(smd))),
        "mmd_rbf": mmd_rbf(Zs, Zt, seed=seed),
        "source_centroid_norm": float(np.linalg.norm(ms)),
    }


def distance_ratio(Zs_norm: np.ndarray, Zt_norm: np.ndarray, Zt_attack: np.ndarray) -> dict:
    """Separability in representation space.

    ``normal_vs_attack`` = mean distance between target-normal and target-attack
    representations; ``intra_normal`` = mean distance within target normals;
    the ratio > 1 indicates attacks sit farther from normal than normal does from
    itself.
    """
    Zs_norm = _subsample(np.asarray(Zs_norm, float), 2000)
    Zt_norm = _subsample(np.asarray(Zt_norm, float), 2000)
    Zt_attack = _subsample(np.asarray(Zt_attack, float), 2000, seed=2)
    mu_s = Zs_norm.mean(axis=0)
    mu_t = Zt_norm.mean(axis=0)
    intra = float(np.linalg.norm(Zt_norm - mu_t, axis=1).mean())
    cross_nn = float(np.linalg.norm(Zt_norm - mu_s, axis=1).mean())
    cross_na = (
        float(np.linalg.norm(Zt_attack - mu_t, axis=1).mean()) if len(Zt_attack) else float("nan")
    )
    return {
        "intra_normal_distance": intra,
        "normal_to_source_centroid": cross_nn,
        "attack_to_target_normal_centroid": cross_na,
        "attack_normal_distance_ratio": (cross_na / intra) if intra > 1e-12 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Supervised linear probe (representation diagnostic only)
# ---------------------------------------------------------------------------
def linear_probe(
    Z_train: np.ndarray, y_train: np.ndarray, Z_test: np.ndarray, y_test: np.ndarray
) -> dict:
    """Logistic-regression probe: does the representation separate attacks at all?

    This is a **representation diagnostic** and must never be reported as a
    zero-shot unsupervised detector. It answers whether separability exists in
    the frozen representation.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score

    Z_train = np.asarray(Z_train, float)
    Z_test = np.asarray(Z_test, float)
    y_train = np.asarray(y_train, int)
    y_test = np.asarray(y_test, int)
    out = {"probe_n_train": int(len(Z_train)), "probe_n_test": int(len(Z_test))}
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        out.update({"probe_roc_auc": float("nan"), "probe_pr_auc": float("nan")})
        return out
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Z_train, y_train)
    s = clf.decision_function(Z_test)
    out["probe_roc_auc"] = float(roc_auc_score(y_test, s))
    out["probe_pr_auc"] = float(average_precision_score(y_test, s))
    return out


__all__ = [
    "apply_standardizer",
    "distance_ratio",
    "fit_centroid",
    "fit_knn_reference",
    "fit_mahalanobis",
    "fit_standardizer",
    "linear_probe",
    "mmd_rbf",
    "representation_shift",
    "score_centroid",
    "score_knn",
    "score_mahalanobis",
]
