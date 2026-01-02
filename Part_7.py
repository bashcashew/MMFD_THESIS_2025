from __future__ import annotations

import argparse, io, json, os, sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    precision_score, recall_score, f1_score, confusion_matrix
)
try:
    from joblib import load
except Exception:
    load = None

#' Optional privacy stem (metrics/privacy.py)
try:
    from metrics.privacy import run as privacy_run
except Exception:
    privacy_run = None

__version__ = "0.8.0"


#' ====== HELPERS ======
def _read_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def _log(out_dir: Path, *msgs: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eval_log.txt", "a", encoding="utf-8") as f:
        for m in msgs:
            f.write(m.rstrip() + "\n")
    print(*msgs, flush=True)

def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))

def _maybe_add_bias(X: np.ndarray, theta_len: int) -> np.ndarray:
    #' If θ is one longer than features, append a 1s bias col to X.
    if X.shape[1] + 1 == theta_len:
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.hstack([X, ones])
    return X

def _load_test(preproc_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str], str, str, pd.DataFrame]:
    schema = _read_json(preproc_dir / "schema.json")
    id_col = schema["id_column"]; label_col = schema["target"]
    df_te = pd.read_parquet(preproc_dir / "test.parquet")
    X_cols = [c for c in df_te.columns if c not in (id_col, label_col)]
    X = df_te[X_cols].to_numpy(dtype=np.float64)
    y = df_te[label_col].to_numpy(dtype=np.int64)
    return X, y, X_cols, id_col, label_col, df_te[[id_col, label_col]].copy()

def _f1_opt_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, Dict[str, float]]:
    p, r, t = precision_recall_curve(y_true, y_prob)
    f1 = np.where((p + r) > 0, (2 * p * r) / (p + r), 0.0)
    idx = int(np.argmax(f1))
    thr = float(t[idx-1]) if idx > 0 and idx-1 < len(t) else 0.5
    return thr, {"best_f1": float(f1[idx]), "precision": float(p[idx]), "recall": float(r[idx])}

def _rank_to_prob(scores: np.ndarray) -> np.ndarray:
    #' Normalize anomaly scores to [0,1] via ranks (IForest)
    ranks = scores.argsort().argsort().astype(np.float64)
    return (ranks / (len(scores) - 1 + 1e-9))


#' ====== MODEL PROBABILITIES ======
def _proba_lr_theta(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xb = _maybe_add_bias(X, theta_len=theta.shape[0])
    return _sigmoid(Xb @ theta)

def _proba_from_model(model: Any, X: np.ndarray, family: str) -> np.ndarray:
    fam = family.lower()
    if fam == "lr":
        #' Centralized LR from Part_3 is a scikit estimator ... use proba
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        if isinstance(model, dict) and "theta" in model:
            theta = np.asarray(model["theta"], dtype=np.float64)
            return _proba_lr_theta(theta, X)
        raise SystemExit("Centralized LR requires a model with predict_proba (from Part_3).")
    if fam in ("rf", "hgb"):
        return model.predict_proba(X)[:, 1]
    if fam == "iforest":
        #' anomaly scores: larger = more anomalous; turn to [0,1]
        s = -model.score_samples(X)
        return _rank_to_prob(s)
    if fam == "gbm_ssrf":
        #' centralized: saved a dict with {"gbm","rf","keep_idx","alpha"} in Part_3
        if isinstance(model, dict) and all(k in model for k in ("gbm","rf","keep_idx","alpha")):
            gbm, rf, keep, a = model["gbm"], model["rf"], np.asarray(model["keep_idx"]), float(model["alpha"])
            return a * gbm.predict_proba(X)[:, 1] + (1.0 - a) * rf.predict_proba(X[:, keep])[:, 1]
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        raise SystemExit("Unsupported GBM-SSRF global model format; expect dict or .predict_proba.")
    raise SystemExit(f"Unknown/unsupported family for proba: {family}")


#' ====== METRICS + COMPUTATION ======
def _compute_metrics(y_true: np.ndarray, p_hat: np.ndarray, thr: float) -> Dict[str, Any]:
    y_pred = (p_hat >= float(thr)).astype(int)
    out = {
        "AUPRC": float(average_precision_score(y_true, p_hat)),
        "ROC_AUC": float(roc_auc_score(y_true, p_hat)) if len(np.unique(y_true)) > 1 else None,
        "F1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0,1]).tolist(),
        "threshold_used": float(thr),
    }
    return out


#' ====== CENTRAL PATHING ======
def eval_centralized(preproc_dir: Path, model_dir: Path, family: str,
                     out_dir: Path, save_preds: bool) -> Dict[str, Any]:
    X, y, X_cols, id_col, label_col, idy = _load_test(preproc_dir)
    #' Load model + threshold
    mdl = load(model_dir / "model.joblib")
    thrp = model_dir / "thresholds.json"
    thr = 0.5
    source = "default_0.5"
    if thrp.exists():
        thr = float(_read_json(thrp).get("threshold", 0.5))
        source = "training_artifact"
    #' Probs
    p = _proba_from_model(mdl, X, family)
    metrics = _compute_metrics(y, p, thr)
    metrics.update({
        "version": __version__,
        "family": family,
        "threshold_source": source,
        "leakage": False
    })
    _write_json(out_dir / "metrics.json", metrics)
    if save_preds:
        pred_df = pd.DataFrame({"id": idy[id_col].astype("string").values,
                                "y_true": y, "p_hat": p, "y_pred": (p >= thr).astype(int)})
        pred_df.to_parquet(out_dir / "predictions.parquet", index=False)
    _log(out_dir, f"[Part_7:{__version__}] centralized | {family} | AUPRC={metrics['AUPRC']:.4f} F1={metrics['F1']:.4f}")
    return metrics


#' ====== FL: server global path ======
def _server_latest_round_dir(server_dir: Path) -> Path:
    rounds = sorted([d for d in (server_dir / "rounds").glob("*") if d.is_dir()])
    if not rounds:
        raise SystemExit(f"No rounds found in {server_dir/'rounds'}")
    return rounds[-1]

def eval_fl_server(server_dir: Path, preproc_dir: Path, family: str,
                   out_dir: Path, tune_on: str, save_preds: bool) -> Dict[str, Any]:
    X, y, X_cols, id_col, label_col, idy = _load_test(preproc_dir)
    rdir = _server_latest_round_dir(server_dir)
    #' Load global
    if family == "lr":
        j = _read_json(rdir / "theta.json")
        theta = np.asarray(j["theta"], dtype=np.float64)
        p = _proba_lr_theta(theta, X)
        thr = 0.5
        source = "default_0.5"
    else:
        mdl_path = rdir / "global_model.joblib"
        if not mdl_path.exists():
            raise SystemExit(f"Global model not found at {mdl_path}")
        mdl = load(mdl_path)
        p = _proba_from_model(mdl, X, family)
        thr = 0.5
        source = "default_0.5"

    leakage = False
    if tune_on.lower() == "test":
        thr, _ = _f1_opt_threshold(y, p)
        leakage = True
        source = "best_F1_on_TEST"

    metrics = _compute_metrics(y, p, thr)
    metrics.update({
        "version": __version__,
        "family": family,
        "threshold_source": source,
        "leakage": leakage,
        "server_round_dir": str(rdir)
    })
    _write_json(out_dir / "metrics.json", metrics)
    if save_preds:
        pred_df = pd.DataFrame({"id": idy[id_col].astype("string").values,
                                "y_true": y, "p_hat": p, "y_pred": (p >= thr).astype(int)})
        pred_df.to_parquet(out_dir / "predictions.parquet", index=False)
    _log(out_dir, f"[Part_7:{__version__}] fl_server | {family} | AUPRC={metrics['AUPRC']:.4f} F1={metrics['F1']:.4f} (round={rdir.name})")
    return metrics


#' ====== FL: per-client aggregation ======
def eval_fl_clients(server_dir: Path, preproc_dirs: List[Path], family: str,
                    out_dir: Path, tune_on: str, save_preds: bool) -> Dict[str, Any]:
    rdir = _server_latest_round_dir(server_dir)
    if family == "lr":
        theta = np.asarray(_read_json(rdir / "theta.json")["theta"], dtype=np.float64)
    else:
        mdl = load(rdir / "global_model.joblib")

    per_client = []
    all_y, all_p, all_ids = [], [], []
    for pdir in preproc_dirs:
        X, y, X_cols, id_col, label_col, idy = _load_test(pdir)
        if family == "lr":
            p = _proba_lr_theta(theta, X)
        else:
            p = _proba_from_model(mdl, X, family)
        #' thresholding chosen consistently
        thr = 0.5
        source = "default_0.5"
        leak = False
        if tune_on.lower() == "test":
            thr, _ = _f1_opt_threshold(y, p); source = "best_F1_on_TEST"; leak = True
        m = _compute_metrics(y, p, thr)
        m.update({"client_dir": str(pdir), "threshold_source": source, "leakage": leak, "n": int(len(y))})
        per_client.append(m)
        all_y.append(y); all_p.append(p); all_ids.append(idy[id_col].astype("string").values)

    Y = np.concatenate(all_y); P = np.concatenate(all_p)
    thr_micro = 0.5
    if tune_on.lower() == "test":
        thr_micro, _ = _f1_opt_threshold(Y, P)
    micro = _compute_metrics(Y, P, thr_micro)
    macro = {
        k: (float(np.nanmean([c[k] for c in per_client if c[k] is not None]))
            if isinstance(per_client[0][k], (int, float)) and k != "threshold_used"
            else None)
        for k in ["AUPRC","ROC_AUC","F1","precision","recall"]
    }

    report = {
        "version": __version__,
        "family": family,
        "server_round_dir": str(rdir),
        "per_client": per_client,
        "micro": {**micro, "threshold_source": ("best_F1_on_TEST" if tune_on.lower()=="test" else "default_0.5"),
                  "leakage": (tune_on.lower()=="test")},
        "macro": macro
    }
    _write_json(out_dir / "metrics.json", report)

    if save_preds:
        ids = np.concatenate(all_ids)
        pred_df = pd.DataFrame({"id": ids,
                                "y_true": Y, "p_hat": P,
                                "y_pred": (P >= micro["threshold_used"]).astype(int)})
        pred_df.to_parquet(out_dir / "predictions.parquet", index=False)

    _log(out_dir, f"[Part_7:{__version__}] fl_clients | {family} | micro AUPRC={report['micro']['AUPRC']:.4f} F1={report['micro']['F1']:.4f}")
    return report


#' ======PRIVACY ======
def _guess_root_from_server_dir(server_dir: Optional[Path]) -> Optional[Path]:
    if not server_dir:
        return None
    try:
        return server_dir.parent.parent
    except Exception:
        return None

def _maybe_path(p: Optional[Path]) -> Optional[Path]:
    return p if (p and p.exists()) else None

def _auto_find_he_dir(root: Optional[Path]) -> Optional[Path]:
    if not root:
        return None
    for cand in [root / "he" / "paillier", root / "he", root]:
        if (cand / "paillier_public.json").exists() or (cand / "public.json").exists():
            return cand / ("paillier" if (cand / "paillier").exists() else "")
    return None

def _auto_find_psi_dir(root: Optional[Path]) -> Optional[Path]:
    if not root:
        return None
    cand = root / "psi"
    return cand if cand.exists() else None

def _collect_updates_from_round(server_dir: Optional[Path]) -> List[Path]:
    if not server_dir:
        return []
    try:
        rdir = _server_latest_round_dir(server_dir)
    except SystemExit:
        return []
    upd = rdir / "updates"
    return list(upd.glob("*.npy")) if upd.exists() else []

def _default_privacy_out(server_dir: Optional[Path], out_dir: Path) -> Path:
    root = _guess_root_from_server_dir(server_dir) or out_dir.parent
    return root / "metrics" / "privacy"

def _maybe_run_privacy(args: argparse.Namespace,
                       out_dir: Path,
                       server_dir: Optional[Path],
                       predictions_path: Optional[Path]) -> None:
    if not getattr(args, "privacy", False):
        return
    if privacy_run is None:
        _log(out_dir, "[privacy] metrics.privacy not importable; skipping.")
        return

    root = _guess_root_from_server_dir(server_dir)
    he_dir = _maybe_path(args.privacy_he_dir) or _auto_find_he_dir(root)
    psi_dir = _maybe_path(args.privacy_psi_dir) or _auto_find_psi_dir(root)
    client_updates = args.privacy_client_updates or _collect_updates_from_round(server_dir)
    k_min = args.privacy_k_min
    unauth = args.privacy_unauth_decrypts
    mi_labels = args.privacy_mi_labels


    pred_path = args.privacy_predictions or predictions_path
    if pred_path and not pred_path.exists():
        pred_path = None

    pout = args.privacy_out or _default_privacy_out(server_dir, out_dir)
    try:
        rep = privacy_run(
            he_dir=he_dir,
            psi_dir=psi_dir,
            out_dir=pout,
            client_update_paths=client_updates if client_updates else None,
            k_min=k_min,
            unauthorized_decrypts=unauth,
            pred_path=pred_path,
            mi_labels=mi_labels,
        )
        _log(out_dir, f"[privacy] wrote {pout/'privacy_eval.json'} (scheme={rep.get('he',{}).get('scheme','na')})")
    except Exception as e:
        _log(out_dir, f"[privacy] ERROR: {e!r} (he_dir={he_dir} psi_dir={psi_dir} updates={len(client_updates)})")


#' ====== CLI ======
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Part_7: Evaluate centralized or FL global models on Part_2 test sets."
    )
    p.add_argument("--mode", choices=["centralized","fl_server","fl_clients"], required=True)

    #' Model family (guides the prob logic)
    p.add_argument("--model", choices=["lr","rf","hgb","gbm_ssrf","iforest"], required=True)

    #' Centralized path
    p.add_argument("--preproc-dir", type=Path, help="Part_2 directory with test.parquet & schema.json.")
    p.add_argument("--model-dir", type=Path, help="Part_3 model directory (centralized).")

    #' FL server/global
    p.add_argument("--server-dir", type=Path, help="Part_4 server out directory (has rounds/...).")
    p.add_argument("--preproc-dirs", type=Path, nargs="*", help="For fl_clients: list of Part_2 dirs, one per client.")

    #' Threshold control
    p.add_argument("--tune-on", choices=["none","test"], default="none",
                   help="If 'test', choose best-F1 threshold on TEST (marked as leakage).")

    #' Output & preds
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--save-preds", action="store_true")

    #' Privacy self-tests
    p.add_argument("--privacy", action="store_true",
                   help="Run privacy/encryption self-tests (metrics/privacy.py) after evaluation.")
    p.add_argument("--privacy-he-dir", type=Path, default=None, help="Override Paillier context dir.")
    p.add_argument("--privacy-psi-dir", type=Path, default=None, help="Override PSI artifact dir.")
    p.add_argument("--privacy-client-updates", type=Path, nargs="*", default=None,
                   help="Override per-client update vectors (npy/csv). If omitted, auto-discovers from latest round.")
    p.add_argument("--privacy-k-min", type=int, default=None, help="Policy: min clients required per decrypt.")
    p.add_argument("--privacy-unauth-decrypts", type=int, default=None, help="Count of decrypts below k_min (should be 0).")
    p.add_argument("--privacy-mi-labels", type=Path, default=None, help="CSV with columns id,is_member (0/1).")
    p.add_argument("--privacy-predictions", type=Path, default=None,
                   help="Explicit predictions.parquet for MI probe; defaults to the file written by this run if --save-preds.")
    p.add_argument("--privacy-out", type=Path, default=None,
                   help="Output dir for privacy_eval.*; defaults to <ROOT>/metrics/privacy where ROOT is inferred from --server-dir.")
    return p

def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = args.out; out_dir.mkdir(parents=True, exist_ok=True)

    predictions_path: Optional[Path] = None

    if args.mode == "centralized":
        if not args.preproc_dir or not args.model_dir:
            raise SystemExit("--preproc-dir and --model-dir are required for centralized mode.")
        eval_centralized(args.preproc_dir, args.model_dir, args.model, out_dir, args.save_preds)

        pp = out_dir / "predictions.parquet"
        predictions_path = pp if (args.save_preds and pp.exists()) else None
        _maybe_run_privacy(args, out_dir, server_dir=None, predictions_path=predictions_path)

    elif args.mode == "fl_server":
        if not args.server_dir or not args.preproc_dir:
            raise SystemExit("--server-dir and --preproc-dir are required for fl_server mode.")
        eval_fl_server(args.server_dir, args.preproc_dir, args.model, out_dir, args.tune_on, args.save_preds)
        pp = out_dir / "predictions.parquet"
        predictions_path = pp if (args.save_preds and pp.exists()) else None
        _maybe_run_privacy(args, out_dir, server_dir=args.server_dir, predictions_path=predictions_path)

    elif args.mode == "fl_clients":
        if not args.server_dir or not args.preproc_dirs:
            raise SystemExit("--server-dir and --preproc-dirs are required for fl_clients mode.")
        eval_fl_clients(args.server_dir, args.preproc_dirs, args.model, out_dir, args.tune_on, args.save_preds)
        pp = out_dir / "predictions.parquet"
        predictions_path = pp if (args.save_preds and pp.exists()) else None
        _maybe_run_privacy(args, out_dir, server_dir=args.server_dir, predictions_path=predictions_path)

    else:
        raise SystemExit(f"Unknown mode: {args.mode}")

    _log(out_dir, "[Part_7] Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
