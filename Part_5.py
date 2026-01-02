from __future__ import annotations

import argparse, base64, io, json, os, sys, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
try:

    from urllib3.util.retry import Retry
    import urllib3
except Exception:
    Retry = None



from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.model_selection import train_test_split

__version__ = "0.7.0"


#' ====== Small IO + logging helpers ======

def _read_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def _log(out_dir: Path, *msgs: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "client_log.txt", "a", encoding="utf-8") as f:
        for m in msgs:
            f.write(m.rstrip() + "\n")
    print(*msgs, flush=True)

def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")

#' ---------------------------------------------------------------------------
#' HTTP / TLS client
#' ---------------------------------------------------------------------------
class FLHttp:
    def __init__(self, base_url: str, *, ca: Optional[Path], cert: Optional[Path], key: Optional[Path], insecure: bool):
        self.base_url = base_url.rstrip("/")
        self.sess = requests.Session()
        if insecure:
            self.sess.verify = False
            if 'urllib3' in globals() and urllib3:
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        else:
            self.sess.verify = str(ca) if ca else True
        if cert and key:
            self.sess.cert = (str(cert), str(key))
        elif cert and not key:
            self.sess.cert = str(cert)

        if Retry is not None:
            retries = Retry(
                total=5,
                backoff_factor=0.5,
                status_forcelist=(502, 503, 504),
                allowed_methods=frozenset(["GET", "POST"]),
                raise_on_status=False,
            )
            adapter = HTTPAdapter(max_retries=retries)
            self.sess.mount("http://", adapter)
            self.sess.mount("https://", adapter)

    def get(self, path: str) -> Dict[str, Any]:
        r = self.sess.get(f"{self.base_url}{path}", timeout=30)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self.sess.post(f"{self.base_url}{path}", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()

#' ---------------------------------------------------------------------------
#' HE adapters (client-side encrypt)
#' ---------------------------------------------------------------------------
class ClientHE:
    def __init__(self, scheme: str, he_dir: Optional[Path]):
        self.scheme = scheme
        self.he_dir = he_dir
        self.impl = None
        if scheme == "none":
            return
        if scheme == "paillier":
            try:
                from crypto.he_paillier import PaillierContext
            except Exception as e:
                raise SystemExit(f"[HE] Paillier selected but crypto.he_paillier not available: {e}")
            self.impl = PaillierContext.load_public(he_dir)
        elif scheme == "ckks":
            try:
                from crypto.he_ckks import CKKSContext
            except Exception as e:
                raise SystemExit(f"[HE] CKKS selected but crypto.he_ckks not available: {e}")
            self.impl = CKKSContext.load_public(he_dir)
        else:
            raise SystemExit(f"Unknown HE scheme: {scheme}")

    def encrypt_numpy_to_bytes(self, arr: np.ndarray) -> bytes:
        if self.scheme == "none":
            raise RuntimeError("encrypt_numpy_to_bytes called with HE=none")
        return self.impl.encrypt_numpy_to_bytes(arr)

#' ---------------------------------------------------------------------------
#' Class weights helpers (alongside Part_2)
#' ---------------------------------------------------------------------------
def load_sample_weights(preproc_dir: Path, y: np.ndarray) -> Optional[np.ndarray]:
    cw = None
    p = preproc_dir / "class_weights.json"
    if p.exists():
        j = _read_json(p)
        cw = {int(float(k)): float(v) for k, v in j.items()}
    else:
        cc = preproc_dir / "class_counts.json"
        if cc.exists():
            jj = _read_json(cc)
            tot = sum(jj.values())
            cw = {int(k): (0.0 if v == 0 else float(tot / v)) for k, v in jj.items()}
    if cw is None:
        return None
    return np.array([float(cw.get(int(lbl), 1.0)) for lbl in y], dtype=np.float64)

#' ---------------------------------------------------------------------------
#' LR gradient
#' ---------------------------------------------------------------------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))

def lr_grad_sum(X: np.ndarray, y: np.ndarray, theta: np.ndarray, w: Optional[np.ndarray]) -> np.ndarray:
    """
    Compute sum of gradients of logistic loss over local samples:
    grad_i = (sigmoid(x_i·θ) - y_i) * x_i  ;  sum over i
    If w is provided, multiply each sample's gradient by w_i.
    """
    pred = sigmoid(X @ theta)
    err = (pred - y)
    if w is not None:
        err = err * w
    #' Sum across samples
    g = X.T @ err
    return g.astype(np.float64)

#' ---------------------------------------------------------------------------
#' GBM-SSRF helper (client-side train)
#' ---------------------------------------------------------------------------
def train_gbm_ssrf_local(
    X: np.ndarray, y: np.ndarray, seed: int, threads: Optional[int],
    topk_ratio: float = 0.5, val_size: float = 0.2
) -> Dict[str, Any]:
    """
    Train HistGB + shallow RF on top-K MI features; tune α by F1 on a val split.
    Returns {"gbm","rf","keep_idx","alpha"}.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=val_size, random_state=seed, stratify=y)


    try:
        from sklearn.feature_selection import mutual_info_classif
        mi = mutual_info_classif(X_tr, y_tr, discrete_features=False, random_state=seed)
        order = np.argsort(mi)[::-1]
    except Exception:
        var = X_tr.var(axis=0)
        order = np.argsort(var)[::-1]
    k = max(1, int(X.shape[1] * topk_ratio))
    keep_idx = order[:k]

    #' models
    gbm = HistGradientBoostingClassifier(
        max_depth=None, learning_rate=0.06, max_iter=400, l2_regularization=1.0,
        early_stopping=True, random_state=seed
    )
    rf = RandomForestClassifier(
        n_estimators=600, max_depth=8, min_samples_leaf=4, max_features="sqrt",
        n_jobs=threads if threads else os.cpu_count(), random_state=seed, class_weight=None
    )

    gbm.fit(X_tr, y_tr)
    rf.fit(X_tr[:, keep_idx], y_tr)

    #' tune alpha
    from sklearn.metrics import precision_recall_curve
    def f1_opt_thr(y_true, p):
        p_, r_, t_ = precision_recall_curve(y_true, p)
        f1 = np.where((p_ + r_) > 0, 2 * p_ * r_ / (p_ + r_), 0.0)
        j = int(np.argmax(f1))
        thr = float(t_[j-1]) if j > 0 and j-1 < len(t_) else 0.5
        return float(f1[j]), thr

    p_gbm = gbm.predict_proba(X_val)[:, 1]
    p_rf  = rf.predict_proba(X_val[:, keep_idx])[:, 1]
    best_a, best_f1 = 0.5, -1.0
    for a in np.linspace(0.0, 1.0, 51):
        f1, _ = f1_opt_thr(y_val, a * p_gbm + (1.0 - a) * p_rf)
        if f1 > best_f1:
            best_f1, best_a = f1, float(a)

    return {"gbm": gbm, "rf": rf, "keep_idx": keep_idx, "alpha": best_a}

#' ---------------------------------------------------------------------------
#' Client Main
#' ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Part_5: HFL client for LR, RF, HGB, GBM-SSRF, IF."
    )
    p.add_argument("--client-id", required=True)
    p.add_argument("--model", choices=["lr","rf","hgb","gbm_ssrf","iforest"], required=True)

    #' Data
    p.add_argument("--preproc-dir", type=Path, required=True, help="Dir with train.parquet + schema.json")
    p.add_argument("--add-bias", action="store_true", help="Augment X with a bias term (LR only).")

    #' Server
    p.add_argument("--server", required=True, help="Base URL, e.g., https://127.0.0.1:8842")
    p.add_argument("--rounds", type=int, default=None, help="Override rounds; else poll server /status.")
    p.add_argument("--poll-sec", type=float, default=1.5, help="Poll interval for /status.")

    #' TLS/mTLS
    p.add_argument("--tls-ca", type=Path, default=None, help="CA cert for server verification.")
    p.add_argument("--tls-cert", type=Path, default=None, help="Client cert (for mTLS).")
    p.add_argument("--tls-key", type=Path, default=None, help="Client key (for mTLS).")
    p.add_argument("--insecure", action="store_true", help="Disable TLS verification (dev only).")

    #' HE for LR
    p.add_argument("--he", choices=["none","paillier","ckks"], default="none")
    p.add_argument("--he-dir", type=Path, default=None, help="Dir with Part_1B HE artifacts (public materials).")

    #' Training attribs
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--diversify", action="store_true", help="Diversify tree/GBM/IF seeds per round.")
    p.add_argument("--val-size", type=float, default=0.2, help="VAL size for GBM-SSRF α tuning.")

    #' Output
    p.add_argument("--out", type=Path, required=True)

    #' Secure aggregation (Part_6)
    p.add_argument("--secureagg-label", type=str, default=None,
                   help="If set, apply Part_6 pairwise mask from artifacts/run/secureagg/<label>/clients/<client-id>/mask.npy")
    p.add_argument("--secureagg-dir", type=Path, default=Path("artifacts/run"),
                   help="Root artifacts dir that contains secureagg/<label>/...")
    return p

def set_thread_env(n: Optional[int]) -> None:
    if n is None:
        return
    for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"]:
        os.environ[k] = str(n)

def load_data(preproc_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str], str, str, Optional[np.ndarray]]:
    schema = _read_json(preproc_dir / "schema.json")
    id_col, label_col = schema["id_column"], schema["target"]
    df_tr = pd.read_parquet(preproc_dir / "train.parquet")

    X_cols = [c for c in df_tr.columns if c not in (id_col, label_col)]
    X = df_tr[X_cols].to_numpy(dtype=np.float64)
    y = df_tr[label_col].to_numpy(dtype=np.int64)

    w = load_sample_weights(preproc_dir, y)
    return X, y, X_cols, id_col, label_col, w

def augment_bias(X: np.ndarray) -> np.ndarray:
    ones = np.ones((X.shape[0], 1), dtype=X.dtype)
    return np.hstack([X, ones])

def register(server: FLHttp, client_id: str, model: str, dim: Optional[int]) -> Dict[str, Any]:
    payload = {"client_id": client_id, "model": model}
    if model == "lr":
        if dim is None or dim <= 0:
            raise SystemExit("LR requires a positive 'dim' to register.")
        payload["dim"] = int(dim)
    return server.post("/register", payload)

def fetch_theta(server: FLHttp, default_dim: int) -> np.ndarray:
    info = server.get("/global")
    if info.get("have_global") and "theta" in info:
        return np.asarray(info["theta"], dtype=np.float64)
    #' default: zeros
    return np.zeros(default_dim, dtype=np.float64)

def serialize_model_to_b64(obj: Any) -> str:
    buf = io.BytesIO()
    dump(obj, buf)
    return _b64(buf.getvalue())

def _maybe_load_mask(secureagg_dir: Path, label: Optional[str], client_id: str, dim: int) -> Optional[np.ndarray]:
    if not label:
        return None
    mpath = secureagg_dir / "secureagg" / label / "clients" / client_id / "mask.npy"
    if not mpath.exists():
        raise SystemExit(f"[secureagg] mask not found at {mpath}. "
                         f"Run Part_6: session-init → dev-gensecrets → client-derive first.")
    m = np.load(mpath)
    if m.ndim != 1 or m.shape[0] != dim:
        raise SystemExit(f"[secureagg] mask dim {m.shape} != expected {dim}. "
                         f"Ensure Part_6 --dim matches LR feature dim (incl. bias if --add-bias).")
    return m.astype(np.float64)

def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = args.out; out_dir.mkdir(parents=True, exist_ok=True)
    set_thread_env(args.threads)

    http = FLHttp(args.server, ca=args.tls_ca, cert=args.tls_cert, key=args.tls_key, insecure=args.insecure)

    X, y, X_cols, id_col, label_col, w = load_data(args.preproc_dir)
    if args.model == "lr" and args.add_bias:
        X = augment_bias(X)

    lr_dim = X.shape[1] if args.model == "lr" else None

    ack = register(http, args.client_id, args.model, lr_dim)
    expected_clients = int(ack.get("expected_clients", 1))
    _log(out_dir, f"[Part_5:{__version__}] Registered as {args.client_id} | model={args.model} | "
                  f"round={ack.get('round')} | expected_clients={expected_clients}")

    rounds_total = args.rounds
    if rounds_total is None:
        st = http.get("/status")
        rounds_total = int(st.get("rounds_total", 1))

    he = None
    if args.model == "lr" and args.he != "none":
        he = ClientHE(args.he, args.he_dir)

    for r in range(1, rounds_total + 1):
        while True:
            st = http.get("/status")
            cur = int(st.get("round", 1))
            if cur == r:
                break
            time.sleep(max(0.2, float(args.poll_sec)))


        np.random.seed(args.seed + (r if args.diversify else 0))

        if args.model == "lr":
            theta = fetch_theta(http, default_dim=lr_dim or X.shape[1])
            if theta.shape[0] != (lr_dim or X.shape[1]):
                raise SystemExit(f"Server theta dim {theta.shape[0]} != local dim {(lr_dim or X.shape[1])}")

            gsum = lr_grad_sum(X, y, theta, w)
            if args.he != "none" and args.secureagg_label:
                raise SystemExit("[secureagg] When HE is enabled, do not apply masks. Use either HE or masks, not both.")
            if args.he == "none" and args.secureagg_label:
                mask = _maybe_load_mask(args.secureagg_dir, args.secureagg_label, args.client_id, gsum.shape[0])
                gsum = gsum + mask
            if args.he == "none":
                payload = {
                    "client_id": args.client_id,
                    "round": r,
                    "model": "lr",
                    "n": int(X.shape[0]),
                    "grad_sum": gsum.astype(float).tolist()
                }
            else:
                ct_bytes = he.encrypt_numpy_to_bytes(gsum)
                payload = {
                    "client_id": args.client_id,
                    "round": r,
                    "model": "lr",
                    "n": int(X.shape[0]),
                    "grad_sum": _b64(ct_bytes)
                }

        elif args.model == "rf":
            mdl = RandomForestClassifier(
                n_estimators=400, max_depth=None, min_samples_leaf=2, max_features="sqrt",
                n_jobs=args.threads if args.threads else os.cpu_count(),
                random_state=args.seed + (r if args.diversify else 0),
                class_weight=None
            )
            mdl.fit(X, y, sample_weight=w)
            model_b64 = serialize_model_to_b64(mdl)
            (out_dir / "models").mkdir(parents=True, exist_ok=True)
            (out_dir / "models" / f"round_{r:04d}.joblib").write_bytes(base64.b64decode(model_b64.encode("ascii")))
            payload = {
                "client_id": args.client_id,
                "round": r,
                "model": "rf",
                "n": int(X.shape[0]),
                "model_b64": model_b64
            }

        elif args.model == "hgb":
            mdl = HistGradientBoostingClassifier(
                max_depth=None, learning_rate=0.06, max_iter=400, l2_regularization=1.0,
                early_stopping=True, random_state=args.seed + (r if args.diversify else 0)
            )
            mdl.fit(X, y, sample_weight=w)
            model_b64 = serialize_model_to_b64(mdl)
            (out_dir / "models").mkdir(parents=True, exist_ok=True)
            (out_dir / "models" / f"round_{r:04d}.joblib").write_bytes(base64.b64decode(model_b64.encode("ascii")))
            payload = {
                "client_id": args.client_id,
                "round": r,
                "model": "hgb",
                "n": int(X.shape[0]),
                "model_b64": model_b64
            }

        elif args.model == "iforest":
            mdl = IsolationForest(
                n_estimators=400, max_samples="auto", contamination="auto",
                n_jobs=args.threads if args.threads else os.cpu_count(),
                random_state=args.seed + (r if args.diversify else 0)
            )
            mdl.fit(X)
            model_b64 = serialize_model_to_b64(mdl)
            (out_dir / "models").mkdir(parents=True, exist_ok=True)
            (out_dir / "models" / f"round_{r:04d}.joblib").write_bytes(base64.b64decode(model_b64.encode("ascii")))
            payload = {
                "client_id": args.client_id,
                "round": r,
                "model": "iforest",
                "n": int(X.shape[0]),
                "model_b64": model_b64
            }

        elif args.model == "gbm_ssrf":
            local = train_gbm_ssrf_local(
                X=X, y=y, seed=args.seed + (r if args.diversify else 0),
                threads=args.threads, topk_ratio=0.5, val_size=float(args.val_size)
            )
            model_b64 = serialize_model_to_b64(local)
            (out_dir / "models").mkdir(parents=True, exist_ok=True)
            (out_dir / "models" / f"round_{r:04d}.joblib").write_bytes(base64.b64decode(model_b64.encode("ascii")))
            payload = {
                "client_id": args.client_id,
                "round": r,
                "model": "gbm_ssrf",
                "n": int(X.shape[0]),
                "model_b64": model_b64
            }

        else:
            raise SystemExit(f"Unknown model: {args.model}")

        ack = http.post("/update", payload)
        _write_json(out_dir / "last_payload.json", {
            "when": _now(), "round": r, "model": args.model,
            "n": int(X.shape[0]), "he": args.he,
            "secureagg_label": args.secureagg_label or ""
        })
        _log(out_dir, f"[Part_5:{__version__}] round={r} uploaded; server says: {ack}")


    _log(out_dir, f"[Part_5:{__version__}] All rounds complete.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
