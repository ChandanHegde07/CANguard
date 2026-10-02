
from __future__ import annotations

import numpy as np
import pandas as pd


def catalog_errors(
    test_df: pd.DataFrame,
    scores: np.ndarray,
    y_pred: np.ndarray,
    feature_cols: list[str],
    top_k: int = 20,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Collect top-K FP and FN with dominant residual features."""
    y = test_df["is_attack"].values.astype(int)
    scores = np.asarray(scores)
    y_pred = np.asarray(y_pred).astype(int)
    if threshold is None:
        threshold = float("nan")

    rows = []
    # False positives: normal predicted attack, highest scores
    fp_mask = (y == 0) & (y_pred == 1)
    fp_idx = np.where(fp_mask)[0]
    if len(fp_idx):
        order = fp_idx[np.argsort(-scores[fp_idx])][:top_k]
        for rank, i in enumerate(order, 1):
            rows.append(_row(test_df, i, scores, threshold, feature_cols, "FP", rank))

    # Also high-score normals below threshold (near-FP) if few FP
    if len(fp_idx) < top_k:
        near = np.where(y == 0)[0]
        near = near[np.argsort(-scores[near])][: top_k - len(fp_idx)]
        for rank, i in enumerate(near, 1):
            if y_pred[i] == 1:
                continue
            rows.append(_row(test_df, i, scores, threshold, feature_cols, "near_FP", rank))

    # False negatives: attack predicted normal, lowest scores
    fn_mask = (y == 1) & (y_pred == 0)
    fn_idx = np.where(fn_mask)[0]
    if len(fn_idx):
        order = fn_idx[np.argsort(scores[fn_idx])][:top_k]
        for rank, i in enumerate(order, 1):
            rows.append(_row(test_df, i, scores, threshold, feature_cols, "FN", rank))

    return pd.DataFrame(rows)


def _row(df, i, scores, threshold, feature_cols, kind, rank):
    row = df.iloc[int(i)]
    feats = {c: float(row[c]) if c in row.index and pd_notna(row[c]) else 0.0 for c in feature_cols}
    # dominant by absolute residual magnitude
    ranked = sorted(feats.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top3 = ranked[:3]
    explanation = _explain(kind, row, top3, float(scores[int(i)]), threshold)
    return {
        "error_type": kind,
        "rank": rank,
        "can_id": row.get("can_id", ""),
        "timestamp": row.get("timestamp", float("nan")),
        "is_attack": int(row.get("is_attack", 0)),
        "score": float(scores[int(i)]),
        "threshold": float(threshold) if threshold == threshold else float("nan"),
        "top_feature_1": top3[0][0] if top3 else "",
        "top_feature_1_value": top3[0][1] if top3 else float("nan"),
        "top_feature_2": top3[1][0] if len(top3) > 1 else "",
        "top_feature_2_value": top3[1][1] if len(top3) > 1 else float("nan"),
        "top_feature_3": top3[2][0] if len(top3) > 2 else "",
        "top_feature_3_value": top3[2][1] if len(top3) > 2 else float("nan"),
        "likely_explanation": explanation,
    }


def pd_notna(x):
    try:
        return x == x and x is not None
    except Exception:
        return False


def _explain(kind, row, top3, score, threshold):
    cid = str(row.get("can_id", ""))
    feats = ",".join(f"{n}={v:.2f}" for n, v in top3[:2])
    if kind == "FN":
        if any("iat" in n for n, _ in top3[:2]):
            return f"missed_attack id={cid}; mild IAT residual ({feats}); score below thr"
        if any("byte" in n for n, _ in top3[:2]):
            return f"missed_attack id={cid}; payload residual weak ({feats})"
        return f"missed_attack id={cid}; score={score:.3f}<thr; dominant={feats}"
    if kind in ("FP", "near_FP"):
        if any("iat" in n for n, _ in top3[:2]):
            return f"benign_iat_shift id={cid}; {feats}"
        if any("byte" in n for n, _ in top3[:2]):
            return f"benign_payload_variation id={cid}; {feats}"
        return f"elevated_score id={cid}; {feats}"
    return "unknown"
