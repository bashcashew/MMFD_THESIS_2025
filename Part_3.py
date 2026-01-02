from __future__ import annotations

import argparse, json, os, sys, math
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest, HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.model_selection import train_test_split

__version__ = "0.7.0"


#' ====== Threads ======
def set_thread_env(n: Optional[int]) -> None:
    if n is None:
        return
    for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"]:
        os.environ[k] = str(n)


#' ====== IO HELP ======
def _read_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_json(p: Path, o: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(o, f, indent=2, sort_keys=True)

def _log(out_dir: Path, *msgs: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train_log.txt", "a", encoding="utf-8") as f:
        for m in msgs:
            f.write(m.rstrip() + "\n")
    print(*msgs, flush=True)


#' ====== WEIGHTS & THRESHOLDS ======
def resolve_class_weights(preproc_dir: Path) -> Optional[Dict[str, float]]:
    """Prefer explicit class_weights.json; else derive inverse_global from class_counts.json; else None."""
    cw = preproc_dir / "class_weights.json"
    cc = preproc_dir / "class_counts.json"
    if cw.exists():
        j = _read_json(cw)
        return {str(int(float(k))): float(v) for k, v in j.items()}
    if cc.exists():
        j = _read_json(cc)
        tot = sum(j.values())
        inv = {k: (0.0 if v == 0 else float(tot / v)) for k, v in j.items()}
        return inv
    return None

def sample_weight_from_class_weights(y: np.ndarray, cw: Optional[Dict[str, float]]) -> Optional[np.ndarray]:
    if cw is None:
        return None
    return np.array([float(cw.get(str(int(lbl)), 1.0)) for lbl in y], dtype=np.float64)

def f1_opt_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    Return the probability threshold that maximizes F1 on (y_true, y_prob),
    using numerically safe division and correct PR/threshold alignment.
    """
    #' PR thresholds `t` aligns with p[1:], r[1:] (sklearn)
    with np.errstate(divide="ignore", invalid="ignore"):
        p, r, t = precision_recall_curve(y_true, y_prob)

    #' Aligns arrays to thresholds
    if t.size == 0:
        #' fallback
        return 0.5, {"best_f1": 0.0, "precision": float(p[-1] if p.size else 0.0),
                     "recall": float(r[-1] if r.size else 0.0)}

    p1 = p[1:]
    r1 = r[1:]
    #' Safe F1: compute only where (p1+r1)>0; else 0
    f1 = np.divide(2.0 * p1 * r1, (p1 + r1),
                   out=np.zeros_like(p1, dtype=float),
                   where=(p1 + r1) > 0.0)

    if f1.size == 0 or not np.isfinite(f1).any():
        return 0.5, {"best_f1": 0.0, "precision": 0.0, "recall": 0.0}

    k = int(np.nanargmax(f1))
    thr = float(t[k])  #' thresholds align to p[1:], r[1:]
    return thr, {"best_f1": float(f1[k]), "precision": float(p1[k]), "recall": float(r1[k])}


#' ====== MODEL CUILDER ======
def build_lr(cw: Optional[Dict[str, float]], seed: int):
    #' passes imbalance through sample_weight during .fit() to avoid double-weighting.
    return LogisticRegression(solver="liblinear", max_iter=1000, random_state=seed, class_weight=None)


def build_rf(cw: Optional[Dict[str, float]], threads: Optional[int], seed: int):
    #' Uses sample_weight during .fit(); keep class_weight=None to prevent compounding.
    return RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2, max_features="sqrt",
        n_jobs=threads if threads else os.cpu_count(), random_state=seed, class_weight=None
    )

def build_hgb(seed: int, *, early_stopping: bool):
    #' Uses sample_weight in .fit; early_stopping on validation split (sklearn built-in)
    return HistGradientBoostingClassifier(
        max_depth=None, learning_rate=0.06, max_iter=400, l2_regularization=1.0,
        early_stopping=early_stopping, random_state=seed
    )

def build_iforest(threads: Optional[int], seed: int):
    return IsolationForest(
        n_estimators=400, max_samples="auto", contamination="auto",
        n_jobs=threads if threads else os.cpu_count(), random_state=seed
    )

def build_ssrf_rf(threads: Optional[int], seed: int, max_depth: int = 8):
    #' SSRF-style: shallower trees, stronger leaf regularization, sqrt features
    return RandomForestClassifier(
        n_estimators=600, max_depth=max_depth, min_samples_leaf=4, max_features="sqrt",
        n_jobs=threads if threads else os.cpu_count(), random_state=seed, class_weight=None
    )


#' ====== GBM-SSRF HELPER ======
def train_gbm_ssrf(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    threads: Optional[int], seed: int, topk_ratio: float = 0.5
) -> Tuple[Dict[str, Any], float]:
    """
    Train HistGB + SSRF-like RF; preselect top-K features by mutual information
    (fallback: variance). Blend probs with α∈[0,1] maximizing F1 on validation.
    Returns (model_dict, alpha).
    """
    #' Feature pre-selecting
    try:
        from sklearn.feature_selection import mutual_info_classif
        mi = mutual_info_classif(X_tr, y_tr, discrete_features=False, random_state=seed)
        order = np.argsort(mi)[::-1]
    except Exception:
        var = X_tr.var(axis=0)
        order = np.argsort(var)[::-1]

    k = max(1, int(X_tr.shape[1] * topk_ratio))
    keep_idx = order[:k]

    gbm = build_hgb(seed=seed, early_stopping=False)
    rf  = build_ssrf_rf(threads=threads, seed=seed, max_depth=8)

    gbm.fit(X_tr, y_tr)
    rf.fit(X_tr[:, keep_idx], y_tr)

    #' probabilities for val
    p_gbm = gbm.predict_proba(X_val)[:, 1]
    p_rf  = rf.predict_proba(X_val[:, keep_idx])[:, 1]

    #' line search α to maximize F1 on validation
    best_alpha, best_f1, best_thr = 0.5, -1.0, 0.5
    for a in np.linspace(0.0, 1.0, 51):
        p_blend = a * p_gbm + (1.0 - a) * p_rf
        thr, stats = f1_opt_threshold(y_val, p_blend)
        if stats["best_f1"] > best_f1:
            best_f1, best_alpha, best_thr = stats["best_f1"], float(a), thr

    model = {"gbm": gbm, "rf": rf, "keep_idx": keep_idx, "alpha": best_alpha, "best_thr": best_thr}
    return model, best_alpha


def predict_gbm_ssrf(model: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    gbm, rf, keep, a = model["gbm"], model["rf"], model["keep_idx"], float(model["alpha"])
    p = a * gbm.predict_proba(X)[:, 1] + (1.0 - a) * rf.predict_proba(X[:, keep])[:, 1]
    return p


#' ====== CLI & MAIN ======
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Part_3: Centralized training on Part_2 features."
    )
    p.add_argument("--preproc-dir", type=Path, required=True, help="Directory with train/test parquet + schema.json.")
    p.add_argument("--model", choices=["lr","rf","hgb","gbm_ssrf","iforest"], required=True)
    p.add_argument("--out", type=Path, required=True, help="Output dir for model + metrics.")
    p.add_argument("--threads", type=int, default=None, help="Thread cap (overrides env if set).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-size", type=float, default=0.2, help="Fraction of TRAIN for validation.")
    return p

def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    set_thread_env(args.threads)

    #' autodetect npy-based artifacts first
    npy_train = args.preproc_dir / "X_train.npy"
    npy_test  = args.preproc_dir / "X_test.npy"
    use_npy = npy_train.exists() and npy_test.exists()

    if use_npy:
        #' NPY layout from preprocessing/preprocessing.py
        X_tr_full = np.load(npy_train).astype(np.float64, copy=False)
        y_tr_full = np.load(args.preproc_dir / "y_train.npy").astype(np.int64, copy=False)
        X_te = np.load(npy_test).astype(np.float64, copy=False)
        y_te = np.load(args.preproc_dir / "y_test.npy").astype(np.int64, copy=False)

        #' Label name (best-effort; not strict downstream)
        label_col = "y"
        yn = args.preproc_dir / "y_name.txt"
        if yn.exists():
            try:
                label_col = (yn.read_text() or "y").strip()
            except Exception:
                pass

        #' We don't have a native ID column; emit a deterministic row index for TEST
        id_col = "row"
        df_te = pd.DataFrame({id_col: np.arange(X_te.shape[0], dtype=np.int64)})

        #' No external CAL in npy layout
        has_cal = False
        df_ca = None

    else:
        #' Legacy Parquet + schema.json path (original behavior)
        schema = _read_json(args.preproc_dir / "schema.json")
        id_col = schema["id_column"]; label_col = schema["target"]

        #' Ensure required files exist
        train_path = args.preproc_dir / "train.parquet"
        test_path  = args.preproc_dir / "test.parquet"
        if not train_path.exists():
            raise SystemExit(f"Missing TRAIN features at: {train_path}")
        if not test_path.exists():
            raise SystemExit(f"Missing TEST features at: {test_path} (enable step-locked TEST or provide external TEST in Part_2)")

        df_tr = pd.read_parquet(train_path)
        df_te = pd.read_parquet(test_path)

        #' Optional CAL (step-locked) for threshold tuning
        cal_path = args.preproc_dir / "cal.parquet"
        has_cal = cal_path.exists()
        df_ca = pd.read_parquet(cal_path) if has_cal else None

        #' Build matrices
        X_cols = [c for c in df_tr.columns if c not in (id_col, label_col)]
        X_tr_full = df_tr[X_cols].to_numpy(dtype=np.float64)
        y_tr_full = df_tr[label_col].to_numpy(dtype=np.int64)
        X_te = df_te[X_cols].to_numpy(dtype=np.float64)
        y_te = df_te[label_col].to_numpy(dtype=np.int64)

    #' ====== VALIDATION SPLIT ======
    if has_cal:
        X_val = df_ca[X_cols].to_numpy(dtype=np.float64)
        y_val = df_ca[label_col].to_numpy(dtype=np.int64)
        X_tr, y_tr = X_tr_full, y_tr_full
        _log(out_dir, "[Part_3] Using external CAL set for threshold tuning.")
    else:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr_full, y_tr_full, test_size=float(args.val_size), random_state=args.seed, stratify=y_tr_full
        )
        _log(out_dir, "[Part_3] Using TRAIN→VAL split for threshold tuning (no CAL found).")

    #' ====== WEIGHTS RESOLUTION ======
    sw_path = args.preproc_dir / "sample_weight_train.npy"
    if sw_path.exists():
        try:
            w_tr = np.load(sw_path).astype(np.float64, copy=False)
            if X_tr.shape[0] != y_tr_full.shape[0]:
                # Since we split "in-memory", recompute indices by matching object identity.
                # Simpler & deterministic: re-create the split to get the mask.
                Xf, Xv, yf, yv, wf, wv = train_test_split(
                    X_tr_full, y_tr_full, w_tr, test_size=float(args.val_size),
                    random_state=args.seed, stratify=y_tr_full
                )
                w_tr = wf
        except Exception:
            w_tr = None
    else:
        cw = resolve_class_weights(args.preproc_dir)
        w_tr = sample_weight_from_class_weights(y_tr, cw)

    #' ====== TRAINING ======
    model_obj = None
    get_proba = None

    if args.model == "lr":
        mdl = build_lr(cw=cw if 'cw' in locals() else None, seed=args.seed)
        mdl.fit(X_tr, y_tr, sample_weight=w_tr)
        def _proba(X): return mdl.predict_proba(X)[:, 1]
        model_obj, get_proba = mdl, _proba

    elif args.model == "rf":
        mdl = build_rf(cw=cw if 'cw' in locals() else None, threads=args.threads, seed=args.seed)
        mdl.fit(X_tr, y_tr, sample_weight=w_tr)
        def _proba(X): return mdl.predict_proba(X)[:, 1]
        model_obj, get_proba = mdl, _proba

    elif args.model == "hgb":
        mdl = build_hgb(seed=args.seed, early_stopping=not has_cal)
        mdl.fit(X_tr, y_tr, sample_weight=w_tr)
        def _proba(X): return mdl.predict_proba(X)[:, 1]
        model_obj, get_proba = mdl, _proba

    elif args.model == "iforest":
        mdl = build_iforest(threads=args.threads, seed=args.seed)
        mdl.fit(X_tr_full)
        def _proba(X):
            s = -mdl.score_samples(X)  #' larger = more anomalous
            ranks = s.argsort().argsort().astype(np.float64)
            return (ranks / (len(s) - 1 + 1e-9))
        model_obj, get_proba = mdl, _proba

    elif args.model == "gbm_ssrf":
        #' Train GBM-SSRF RF; tune α on val
        model_dict, alpha = train_gbm_ssrf(X_tr, y_tr, X_val, y_val, args.threads, args.seed)
        def _proba(X): return predict_gbm_ssrf(model_dict, X)
        model_obj, get_proba = model_dict, _proba

    else:
        raise SystemExit(f"Unknown model: {args.model}")

    #' Threshold tuning on VAL
    p_val = get_proba(X_val)
    thr, thr_stats = f1_opt_threshold(y_val, p_val)

    #' Evaluate on TEST
    p_te = get_proba(X_te)
    y_hat = (p_te >= thr).astype(int)

    metrics = {
        "version": __version__,
        "model": args.model,
        "AUPRC": float(average_precision_score(y_te, p_te)),
        "ROC_AUC": float(roc_auc_score(y_te, p_te)) if len(np.unique(y_te)) > 1 else None,
        "F1": float(f1_score(y_te, y_hat)),
        "precision": float(precision_score(y_te, y_hat, zero_division=0)),
        "recall": float(recall_score(y_te, y_hat, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_te, y_hat, labels=[0,1]).tolist(),
        "threshold": float(thr),
        "threshold_selection": {"method": "PR-F1 grid", **thr_stats},
    }
    if args.model == "gbm_ssrf":
        metrics["gbm_ssrf"] = {"alpha": float(model_obj["alpha"]), "topk_features": int(len(model_obj["keep_idx"]))}

    #' Persistance
    dump(model_obj, out_dir / "model.joblib")
    _write_json(out_dir / "metrics.json", metrics)
    _write_json(out_dir / "thresholds.json", {"threshold": float(thr), "method": "F1 on validation"})

    #' Save test predictions (actual ID column name for consistency)
    pred_df = pd.DataFrame({
        id_col: df_te[id_col].astype("string").values,
        "y_true": y_te,
        "p_hat": p_te,
        "y_pred": y_hat
    })
    pred_df.to_parquet(out_dir / "predictions.parquet", index=False)

    _log(out_dir, f"[Part_3:{__version__}] {args.model} | "
                 f"AUPRC={metrics['AUPRC']:.4f} ROC_AUC={metrics['ROC_AUC'] if metrics['ROC_AUC'] else 'NA'} "
                 f"F1={metrics['F1']:.4f} P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
                 f"thr={metrics['threshold']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
