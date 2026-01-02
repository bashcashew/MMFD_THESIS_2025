from __future__ import annotations

import argparse
import json
import os
import sys
import re
import hashlib
import hmac as _hmac
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

__version__ = "0.7.5"

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import polars as pl 
except Exception:
    pl = None

import numpy as np


try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:
    pa = None
    pq = None


#' ====== IO UTILITIES ======
def _read_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def _write_lines(p: Path, lines: List[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln.strip() + "\n")

def _ensure_out_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _log(out_dir: Path, *msgs: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "logs.txt", "a", encoding="utf-8") as f:
        for m in msgs:
            f.write(m.rstrip() + "\n")
    print(*msgs, flush=True)


#' ====== PSI HELPERZ ======
def base64_b64decode(s: str) -> bytes:
    import base64
    return base64.b64decode(s)

def _load_hmac_key_salt(psi_dir: Path) -> Tuple[bytes, Optional[bytes]]:
    key_path = psi_dir / "hmac_key.json"
    if not key_path.exists():
        #' Backward-compat: raw files hmac.key/hmac.salt
        key_file = psi_dir / "hmac.key"
        if key_file.exists():
            key = key_file.read_bytes()
            salt = (psi_dir / "hmac.salt").read_bytes() if (psi_dir / "hmac.salt").exists() else None
            return key, salt
        raise SystemExit(f"[PSI:HMAC] Missing key at {key_path} (run Part_1A psi-hmac-init)")
    keyobj = _read_json(key_path)
    key = base64_b64decode(keyobj.get("k", ""))
    salt_b64 = keyobj.get("salt")
    salt = base64_b64decode(salt_b64) if salt_b64 else None
    if not key:
        raise SystemExit("[PSI:HMAC] hmac_key.json missing field 'k' (base64).")
    return key, salt

def _canonify(s: str, mode: str) -> str:
    if s is None:
        return ""
    if mode == "none":
        return s
    if mode == "lower":
        return s.lower()
    if mode == "upper":
        return s.upper()
    if mode == "strip":
        return s.strip()
    if mode == "alnum_lower":
        return re.sub(r"[^0-9a-zA-Z]+", "", s).lower()
    return s

def _hmac_hex(key: bytes, salt: Optional[bytes], s: str) -> str:
    msg = (salt or b"") + s.encode("utf-8", "ignore")
    return _hmac.new(key, msg, hashlib.sha256).hexdigest()

def _filter_with_hmac(
    df: pd.DataFrame,
    id_col: str,
    psi_dir: Path,
    intersection_path: Path,
    canon: Optional[str] = None,
) -> pd.DataFrame:
    """
    Filter df to rows whose HMAC(id_col) is present in the intersection file.
    If canon in {"lower","lowercase","ci","casefold"}, compare case-insensitively.
    """
    key, salt = _load_hmac_key_salt(psi_dir)

    #' read intersection tokens
    keep = {ln.strip() for ln in open(intersection_path, "r", encoding="utf-8") if ln.strip()}
    do_lower = bool(canon and canon.lower() in {"lower", "lowercase", "ci", "casefold"})
    if do_lower:
        keep = {k.lower() for k in keep}

    def _tok(s: Any) -> str:
        h = _hmac_hex(key, salt, str(s))
        return h.lower() if do_lower else h

    toks = df[id_col].astype(str).map(_tok)
    return df.loc[toks.isin(keep)].copy()

def _filter_with_oprf(df: pd.DataFrame, id_col: str, client_tokens_path: Path, intersection_path: Path) -> pd.DataFrame:
    client_tokens = [ln.strip() for ln in open(client_tokens_path, "r", encoding="utf-8") if ln.strip()]
    if len(client_tokens) != len(df):
        raise SystemExit("[PSI:OPRF] Tokens count != dataframe rows; ensure per-row tokens aligned with this file.")
    inter = set(ln.strip() for ln in open(intersection_path, "r", encoding="utf-8") if ln.strip())
    mask = [tok in inter for tok in client_tokens]
    return df.loc[mask].copy()


#' ====== WEIGHTING ======
def _class_counts(y: pd.Series) -> Dict[str, int]:
    vc = y.value_counts(dropna=False)
    return {str(k): int(v) for k, v in vc.items()}

def _balanced_weights(y: pd.Series) -> Dict[str, float]:
    n = len(y)
    cc = y.value_counts()
    k = len(cc)
    return {str(c): float(n / (k * cc[c])) for c in cc.index}

def _inverse_global_weights(global_counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(global_counts.values()) or 1
    return {cls: (0.0 if cnt == 0 else float(total / cnt)) for cls, cnt in global_counts.items()}


#' ====== FEATURE ENGINERRING (PAYSOM PRESET) ======
PAYSIM_REQUIRED = {
    "step","type","amount","nameOrig","oldbalanceOrg","newbalanceOrig",
    "nameDest","oldbalanceDest","newbalanceDest"
}
PAYSIM_OPTIONAL_DROP = {"isFlaggedFraud"}
PAYSIM_DEFAULT_LABEL = "isFraud"
PAYSIM_ID_COLS_ALL = ["nameOrig","nameDest"]  #' hashed if --save-ids (auditing)

def _assert_paysim_schema(df: pd.DataFrame, label: str) -> None:
    missing = [c for c in PAYSIM_REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"[Part_2] PaySim preset: missing required columns: {missing}")
    if label not in df.columns:
        lower_map = {c.lower(): c for c in df.columns}
        if label.lower() in lower_map:
            df.rename(columns={lower_map[label.lower()]: label}, inplace=True)
        else:
            raise SystemExit(f"[Part_2] PaySim preset expects target column '{label}'.")

def _apply_preset(df: pd.DataFrame, preset: str, id_col: str, label_col: str) -> pd.DataFrame:
    if preset == "paysim":
        num_cols = ["step","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "type" in df.columns:
            df["type"] = df["type"].astype("category")
        if label_col in df.columns:
            df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
        drop_cols = [c for c in PAYSIM_OPTIONAL_DROP if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        return df
    return df


#' ====== BASELINE TRANSFORMATIONS ======
def _fit_ohe(train_cat: pd.DataFrame) -> Dict[str, List[str]]:
    cats = {}
    for c in train_cat.columns:
        cats[c] = sorted([str(v) for v in train_cat[c].dropna().unique().tolist()])
    return cats

def _transform_ohe(df_cat: pd.DataFrame, ohe_map: Dict[str, List[str]]) -> pd.DataFrame:
    parts = []
    for c, levels in ohe_map.items():
        col = df_cat[c].astype("string")
        for lv in levels:
            parts.append((f"{c}__{lv}", (col == lv).astype("int8")))
    return pd.concat({n: s for n, s in parts}, axis=1) if parts else pd.DataFrame(index=df_cat.index)

def _fit_scale(train_num: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
    mean = {c: float(train_num[c].mean()) for c in train_num.columns}
    std = {c: float(train_num[c].std(ddof=0) or 1.0) for c in train_num.columns}
    return mean, std

def _apply_scale(df_num: pd.DataFrame, mean: Dict[str, float], std: Dict[str, float]) -> pd.DataFrame:
    out = df_num.copy()
    for c in out.columns:
        denom = std.get(c, 1.0) or 1.0
        out[c] = (out[c] - mean.get(c, 0.0)) / denom
    return out

def _split_types(df: pd.DataFrame, id_col: str, label_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str], pd.Series]:
    cols = [c for c in df.columns if c not in {id_col, label_col}]
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    Xn = df[num].copy() if num else pd.DataFrame(index=df.index)
    Xc = df[cat].copy() if cat else pd.DataFrame(index=df.index)
    y = df[label_col].astype(int if df[label_col].dropna().isin([0,1]).all() else "int64")
    return Xn, Xc, num, cat, y


#' ====== RECIPE ======
def _load_recipe_config(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    try:
        if path.suffix.lower() in {".yml", ".yaml"}:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        raise SystemExit(f"[recipe-config] Failed to read {path}: {e}")

def _use_recipe_fit_transform(fe, df_tr: pd.DataFrame, df_cal: Optional[pd.DataFrame],
                              df_te: Optional[pd.DataFrame], id_col: str, label_col: str):
    fe.fit(df_tr, id_col=id_col, label_col=label_col)
    X_train = fe.transform(df_tr, id_col=id_col, label_col=label_col)
    X_cal = fe.transform(df_cal, id_col=id_col, label_col=label_col) if df_cal is not None else None
    X_test = fe.transform(df_te, id_col=id_col, label_col=label_col) if df_te is not None else None
    return X_train, X_cal, X_test


#' ====== ID HASHES ======
def _load_or_create_hmac_key(out_dir: Path, keyfile: Optional[str]) -> bytes:
    if keyfile:
        return Path(keyfile).read_bytes()
    key_path = out_dir / "hmac_key.bin"
    if key_path.exists():
        return key_path.read_bytes()
    import secrets
    key = secrets.token_bytes(32)
    key_path.write_bytes(key)
    return key

def _hmac_hash_series(series, key: bytes):
    def _h(s):
        b = str(s).encode("utf-8", "ignore")
        return _hmac.new(key, b, hashlib.sha256).hexdigest()
    return series.map(_h)

def _maybe_hash_ids(df: pd.DataFrame, out_dir: Path, save_ids: bool, keyfile: Optional[str]) -> None:
    if not save_ids:
        return
    key = _load_or_create_hmac_key(out_dir, keyfile)
    maps = []
    for col in PAYSIM_ID_COLS_ALL:
        if col in df.columns:
            hcol = f"{col}_hmac"
            df[hcol] = _hmac_hash_series(df[col], key)
            m = df[[col, hcol]].drop_duplicates()
            m["column"] = col
            maps.append(m)
    if maps:
        mp = pd.concat(maps, axis=0, ignore_index=True)
        mp.to_csv(out_dir / "ids_hmac_map.csv", index=False)
        if not keyfile:
            print(f"[Part_2] HMAC key generated at: {out_dir/'hmac_key.bin'}", file=sys.stderr)

def _maybe_hash_ids_chunked(df: pd.DataFrame, out_dir: Path, save_ids: bool, keyfile: Optional[str]) -> None:
    """Chunk-friendly variant appending to ids_hmac_map.csv."""
    if not save_ids:
        return
    key = _load_or_create_hmac_key(out_dir, keyfile)
    maps = []
    for col in PAYSIM_ID_COLS_ALL:
        if col in df.columns:
            hcol = f"{col}_hmac"
            df[hcol] = _hmac_hash_series(df[col], key)
            m = df[[col, hcol]].drop_duplicates()
            m["column"] = col
            maps.append(m)
    if maps:
        mp = pd.concat(maps, axis=0, ignore_index=True)
        path = out_dir / "ids_hmac_map.csv"
        mp.to_csv(path, index=False, mode="a", header=not path.exists())


#' ====== READ and WRITING ======
def _read_frame(path: Path, engine: str = "pandas") -> pd.DataFrame:
    if pd is None:
        raise SystemExit("pandas is required for Part_2.")
    if engine == "polars" and pl is not None:
        try:
            if path.suffix.lower() in {".parquet", ".pq"}:
                return pl.read_parquet(path).to_pandas()
            return pl.read_csv(path).to_pandas()
        except Exception as e:
            print(f"[Part_2] polars read failed ({e}); retrying with pandas…", file=sys.stderr)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)

def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def _normalize_weighting(alias: str) -> str:
    return "class_balanced" if alias == "balanced" else alias


#' ====== STEP-LOCKE ======
def _coerce_numeric_step(df: pd.DataFrame, step_col: str) -> pd.Series:
    if step_col not in df.columns:
        raise SystemExit(f"[step-locked] step column '{step_col}' not found.")
    s = pd.to_numeric(df[step_col], errors="coerce")
    if s.isna().any():
        n_bad = int(s.isna().sum())
        raise SystemExit(f"[step-locked] {n_bad} rows have non-numeric '{step_col}'.")
    return s

def _split_windows_by_step(df: pd.DataFrame, step_col: str,
                           calib_frac: float, test_frac: float,
                           embargo_steps: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if not (0.0 <= calib_frac < 1.0 and 0.0 <= test_frac < 1.0 and calib_frac + test_frac < 1.0):
        raise SystemExit("[step-locked] require 0<=calib<1, 0<=test<1, calib+test<1")

    s = _coerce_numeric_step(df, step_col)
    q_train_end = float(np.quantile(s.to_numpy(), 1.0 - (calib_frac + test_frac)))
    q_cal_end   = float(np.quantile(s.to_numpy(), 1.0 - test_frac)) if test_frac > 0 else float(s.max())

    train_mask = s <= q_train_end
    cal_mask   = (s > (q_train_end + embargo_steps)) & (s <= q_cal_end)
    test_mask  = s > (q_cal_end + embargo_steps)

    df_tr = df.loc[train_mask].copy()
    df_cal = df.loc[cal_mask].copy()
    df_te = df.loc[test_mask].copy()

    meta = {
        "enabled": True,
        "step_col": step_col,
        "fractions": {"train": 1.0 - calib_frac - test_frac, "calib": calib_frac, "test": test_frac},
        "embargo_steps": int(embargo_steps),
        "cutoffs": {"train_end": q_train_end, "cal_end": q_cal_end},
        "counts": {"train": int(len(df_tr)), "calib": int(len(df_cal)), "test": int(len(df_te))}
    }
    return df_tr, df_cal, df_te, meta

def _write_meta_csv(df: pd.DataFrame, id_col: str, step_col: str, path: Path, window_name: str) -> None:
    if id_col not in df.columns or step_col not in df.columns:
        return
    m = df[[id_col, step_col]].copy()
    m["window"] = window_name
    header = not path.exists()
    m.to_csv(path, index=False, mode="a", header=header)


#' ====== STEP-locked under streaming ======
def _compute_cutoffs_from_csv(path: Path, step_col: str, calib_frac: float, test_frac: float,
                              float_dtype: str, chunksize: int) -> Tuple[float, float]:
    if pd is None:
        raise SystemExit("pandas required for streaming.")
    vals = []
    for chunk in pd.read_csv(path, usecols=[step_col], dtype={step_col: float_dtype},
                             chunksize=chunksize):
        vals.append(chunk[step_col].to_numpy())
    s = np.concatenate(vals, axis=0) if vals else np.array([], dtype=np.float32)
    if s.size == 0:
        raise SystemExit("[step-locked] no values found in step column.")
    q_train_end = float(np.quantile(s, 1.0 - (calib_frac + test_frac)))
    q_cal_end   = float(np.quantile(s, 1.0 - test_frac)) if test_frac > 0 else float(np.max(s))
    return q_train_end, q_cal_end


#' ====== CLI ======
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="Part_2_preprocess.py",
        description="PSI-aware preprocessing with presets, baseline transforms, STEP-locked splits, and optional streaming."
    )
    #' Manifest (optional, values can override defaults)
    p.add_argument("--manifest", type=Path, default=None, help="Part_0 run_manifest.json (optional).")

    #' Data
    p.add_argument("--input-train", type=Path, required=True, help="TRAIN CSV/Parquet.")
    p.add_argument("--input-test", type=Path, default=None, help="TEST CSV/Parquet (optional).")

    #' Preset & schema hints
    p.add_argument("--preset", choices=["none","paysim"], default="none")
    p.add_argument("--label-col", type=str, required=False, default=None)
    p.add_argument("--id-col", type=str, required=False, default=None)

    #' Recipe
    p.add_argument("--recipe", choices=["none","research-grade"], default="none")
    p.add_argument("--recipe-config", type=Path, default=None, help="YAML/JSON config (optional).")

    #' PSI
    p.add_argument("--psi-mode", choices=["none","hmac","oprf"], default="none")
    p.add_argument("--psi-intersection", type=Path, default=None, help="intersection_tokens.txt")
    p.add_argument("--psi-hmac-dir", type=Path, default=None, help="Dir with hmac_key.json (HMAC).")
    p.add_argument("--psi-client-tokens", type=Path, default=None, help="Per-row tokens (OPRF).")
    #' Canonicalization to match Part_1A psi-hmac-generate --canon (default: lower)
    p.add_argument("--psi-hmac-canon", choices=["none","lower","upper","strip","alnum_lower"],
                   default="lower", help="Canonicalization for HMAC inputs; must match Part_1A.")

    #
    #' Class imbalance (primary) + alias parity
    p.add_argument("--class-weighting",
                   choices=["none","balanced","class_balanced","inverse_global"],
                   default="none")
    p.add_argument("--weighting",
                   choices=["none","balanced","class_balanced","inverse_global"],
                   help="Alias for --class-weighting")

    #' Engine and ID hashing toggles
    p.add_argument("--engine", choices=["pandas","polars"], default="pandas")
    p.add_argument("--save-ids", action="store_true", help="HMAC-hash ID columns and emit ids_hmac_map.csv")
    p.add_argument("--hmac-keyfile", help="Path to a 32-byte key to use for ID hashing.")

    #' STEP-locked
    p.add_argument("--step-col", type=str, default=None, help="Column name for chronological split (e.g., 'step').")
    p.add_argument("--calib-frac", type=float, default=0.0, help="Fraction of rows for calibration (latest).")
    p.add_argument("--test-frac", type=float, default=0.0, help="Fraction of rows for test (latest).")
    p.add_argument("--embargo-steps", type=int, default=0, help="Gap between windows to reduce bleed.")

    #' Streaming controls
    p.add_argument("--stream", choices=["auto","on","off"], default="auto",
                   help="Stream large CSVs in two passes to keep memory small.")
    p.add_argument("--chunksize", type=int, default=400000, help="CSV chunk size for streaming.")
    p.add_argument("--chunk-rows", type=int, default=None, help="Alias for --chunksize.")
    p.add_argument("--float-dtype", choices=["float32","float64"], default="float32",
                   help="Numeric dtype used during streaming.")
    #' Outputs
    p.add_argument("--emit-stats", action="store_true", help="Print schema/weights after write.")
    p.add_argument("--out", type=Path, required=True, help="Output directory for artifacts.")
    return p


#' ====== STREMING: helpers + core ======
def _should_stream(args, input_is_csv: bool) -> bool:
    #' align alias
    if getattr(args, "chunk_rows", None):
        args.chunksize = args.chunk_rows
    if args.stream == "on":
        return input_is_csv
    if args.stream == "off":
        return False
    #' BKND: stream if CSV and file is large (> 200MB)
    if not input_is_csv:
        return False
    try:
        return (args.input_train.stat().st_size or 0) > 200 * 1024 * 1024
    except Exception:
        return True 

def _paysim_csv_dtypes(float_dtype: str) -> Dict[str, str]:
    return {
        "step": float_dtype,
        "type": "string",
        "amount": float_dtype,
        "nameOrig": "string",
        "oldbalanceOrg": float_dtype,
        "newbalanceOrig": float_dtype,
        "nameDest": "string",
        "oldbalanceDest": float_dtype,
        "newbalanceDest": float_dtype,
        "isFraud": "int8",
        "isFlaggedFraud": "int8",
    }

def _window_mask_from_cutoffs(step_series: pd.Series, train_end: float, cal_end: float, embargo_steps: int):
    s = step_series.astype(float)
    train_mask = s <= train_end
    cal_mask   = (s > (train_end + embargo_steps)) & (s <= cal_end)
    test_mask  = s > (cal_end + embargo_steps)
    return train_mask, cal_mask, test_mask

def _parquet_writers(out_dir: Path):
    writers = {"train": None, "cal": None, "test": None}
    return writers

def _append_table(writers: Dict[str, Optional[pq.ParquetWriter]],
                  key: str, df_out: pd.DataFrame, path: Path):
    if df_out is None or df_out.empty:
        return
    table = pa.Table.from_pandas(df_out, preserve_index=False)
    p = {"train": path / "train.parquet", "cal": path / "cal.parquet", "test": path / "test.parquet"}[key]
    if writers[key] is None:
        writers[key] = pq.ParquetWriter(p, table.schema)
    writers[key].write_table(table)

def _close_writers(writers: Dict[str, Optional[pq.ParquetWriter]]):
    for k, w in writers.items():
        if w is not None:
            w.close()

def _streaming_process(args, out_dir: Path, preset: str, label_col: str, id_col: str,
                       step_col: Optional[str], calib_frac: float, test_frac: float,
                       embargo_steps: int, float_dtype: str, chunksize: int) -> Dict[str, Any]:
    if pa is None or pq is None:
        raise SystemExit("[stream] pyarrow is required for streaming writes (pip install pyarrow).")
    if args.psi_mode != "none":
        raise SystemExit("[stream] PSI modes are not supported in streaming; run with --stream off.")

    #' Determine CSV vs Parquet
    input_is_csv = args.input_train.suffix.lower() not in {".parquet", ".pq"}
    dtypes = _paysim_csv_dtypes(float_dtype) if (preset == "paysim" and input_is_csv) else None

    #' STEP cutoffs
    time_meta: Dict[str, Any] = {"enabled": False}
    if step_col and (calib_frac > 0.0 or (test_frac > 0.0 and args.input_test is None)):
        train_end, cal_end = _compute_cutoffs_from_csv(args.input_train, step_col, calib_frac, test_frac,
                                                       dtypes["step"] if dtypes else float_dtype, chunksize)
        time_meta = {
            "enabled": True,
            "step_col": step_col,
            "fractions": {"train": 1.0 - calib_frac - test_frac, "calib": calib_frac, "test": test_frac},
            "embargo_steps": int(embargo_steps),
            "cutoffs": {"train_end": float(train_end), "cal_end": float(cal_end)},
        }
        _log(out_dir, f"[step-locked:stream] cutoffs(train_end={train_end:.3f}, cal_end={cal_end:.3f}) embargo={embargo_steps}")
    else:
        train_end = cal_end = None

    #' Stats train
    if preset == "paysim":
        num_cols = ["step","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"]
        cat_cols = ["type"]
    else:
        num_cols, cat_cols = [], []
    means = {c: 0.0 for c in num_cols}
    m2 = {c: 0.0 for c in num_cols}
    n_count = {c: 0 for c in num_cols}
    ohe_vocab: Dict[str, set] = {c: set() for c in cat_cols}

    for chunk in pd.read_csv(args.input_train, chunksize=chunksize, dtype=dtypes, low_memory=False):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        if preset == "paysim":
            _assert_paysim_schema(chunk, label_col)
        chunk = _apply_preset(chunk, preset, id_col=id_col, label_col=label_col)
        if time_meta["enabled"]:
            tr_mask, _, _ = _window_mask_from_cutoffs(chunk[step_col], train_end, cal_end, embargo_steps)
            chunk = chunk.loc[tr_mask]
        if chunk.empty:
            continue
        for c in [col for col in num_cols if col in chunk.columns]:
            x = pd.to_numeric(chunk[c], errors="coerce").fillna(0.0).astype(float_dtype)
            arr = x.to_numpy()
            for val in arr:
                n_old = n_count[c]
                n_count[c] = n_old + 1
                delta = float(val) - means[c]
                means[c] += delta / n_count[c]
                m2[c] += delta * (float(val) - means[c])
        for c in [col for col in cat_cols if col in chunk.columns]:
            ohe_vocab[c].update([str(v) for v in chunk[c].dropna().unique().tolist()])

    stds = {c: (max(1e-12, (m2[c] / n_count[c]) ** 0.5) if n_count[c] > 1 else 1.0) for c in num_cols}
    ohe_map = {c: sorted(list(v)) for c, v in ohe_vocab.items()}

    #' Transform + write
    writers = _parquet_writers(out_dir)
    train_counts = {0: 0, 1: 0}

    def _transform_slice(df_slice: pd.DataFrame) -> pd.DataFrame:
        if df_slice is None or df_slice.empty:
            return pd.DataFrame()
        Xn = pd.DataFrame(index=df_slice.index)
        for c in [col for col in num_cols if col in df_slice.columns]:
            x = pd.to_numeric(df_slice[c], errors="coerce").fillna(0.0).astype(float_dtype)
            Xn[c] = (x - means[c]) / (stds[c] if stds[c] != 0 else 1.0)
        Xc = pd.DataFrame(index=df_slice.index)
        for c in [col for col in cat_cols if col in df_slice.columns]:
            col = df_slice[c].astype("string")
            for lv in ohe_map.get(c, []):
                Xc[f"{c}__{lv}"] = (col == lv).astype("int8")
        X = pd.concat([Xn, Xc], axis=1)
        X.insert(0, id_col, df_slice[id_col].astype("string"))
        X[label_col] = pd.to_numeric(df_slice[label_col], errors="coerce").fillna(0).astype("int8").values
        return X

    for chunk in pd.read_csv(args.input_train, chunksize=chunksize, dtype=dtypes, low_memory=False):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        chunk = _apply_preset(chunk, preset, id_col=id_col, label_col=label_col)
        if chunk.empty:
            continue

        if time_meta["enabled"]:
            tr_mask, ca_mask, te_mask = _window_mask_from_cutoffs(chunk[step_col], train_end, cal_end, embargo_steps)
        else:
            tr_mask = pd.Series(True, index=chunk.index)
            ca_mask = pd.Series(False, index=chunk.index)
            te_mask = pd.Series(False, index=chunk.index)

        df_tr = chunk.loc[tr_mask]
        X_tr = _transform_slice(df_tr)
        if not X_tr.empty:
            _append_table(writers, "train", X_tr, out_dir)
            vals, cnts = np.unique(X_tr[label_col].to_numpy(), return_counts=True)
            for v, c in zip(vals.tolist(), cnts.tolist()):
                if int(v) in train_counts:
                    train_counts[int(v)] += int(c)

        df_ca = chunk.loc[ca_mask]
        X_ca = _transform_slice(df_ca)
        if not X_ca.empty:
            _append_table(writers, "cal", X_ca, out_dir)

        if args.input_test is None:
            df_te = chunk.loc[te_mask]
            X_te = _transform_slice(df_te)
            if not X_te.empty:
                _append_table(writers, "test", X_te, out_dir)

        #' Append ID map + metadata
        _maybe_hash_ids_chunked(chunk[[c for c in PAYSIM_ID_COLS_ALL if c in chunk.columns]], out_dir, args.save_ids, args.hmac_keyfile)
        if step_col:
            if not df_tr.empty:
                _write_meta_csv(df_tr, id_col, step_col, out_dir / "meta_train.csv", "train")
            if not df_ca.empty:
                _write_meta_csv(df_ca, id_col, step_col, out_dir / "meta_cal.csv", "cal")
            if args.input_test is None and not df_te.empty:
                _write_meta_csv(df_te, id_col, step_col, out_dir / "meta_test.csv", "test")

    _close_writers(writers)

    #' weighting artifacts from TRAIN
    cw = _normalize_weighting(getattr(args, "weighting", None) or args.class_weighting)
    if (train_counts[0] + train_counts[1]) > 0:
        if cw == "class_balanced":
            N = float(train_counts[0] + train_counts[1]); k = 2.0
            _write_json(out_dir / "class_weights.json",
                        {"0": float(N / (k * max(1, train_counts[0]))),
                         "1": float(N / (k * max(1, train_counts[1])))})
        elif cw == "inverse_global":
            _write_json(out_dir / "class_counts.json",
                        {"0": int(train_counts[0]), "1": int(train_counts[1])})

    #' schema
    schema = {
        "version": __version__,
        "preset": preset,
        "recipe": {"name": "none", "config": None},
        "target": label_col,
        "id_column": id_col,
        "time_splits": time_meta,
        "numeric": ["step","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"] if preset == "paysim" else [],
        "categorical": ["type"] if preset == "paysim" else [],
        "ohe_categories": {k: v for k, v in ohe_map.items()},
        "scaler": {"mean": {c: float(means[c]) for c in means},
                   "std":  {c: float(stds[c])  for c in stds}},
    }
    _write_json(out_dir / "schema.json", schema)

    #' logging
    cal_rows = 0
    if (out_dir / "cal.parquet").exists():
        try:
            cal_rows = pq.ParquetFile(out_dir / "cal.parquet").metadata.num_rows
        except Exception:
            pass
    te_rows = 0
    if args.input_test is None and (out_dir / "test.parquet").exists():
        try:
            te_rows = pq.ParquetFile(out_dir / "test.parquet").metadata.num_rows
        except Exception:
            pass
    tr_rows = train_counts[0] + train_counts[1]
    _log(out_dir, f"[Part_2:{__version__} stream] Done. Rows(train)={tr_rows} Rows(cal)={cal_rows} Rows(test)={te_rows} "
                  f"| class_weighting={cw}")
    return schema


#ø ====== MAIN ======
def _resolve_defaults(args) -> Tuple[str, str, str, str, Optional[Path]]:
    preset = args.preset
    recipe = args.recipe
    recipe_cfg_path = args.recipe_config
    label = args.label_col
    idc = args.id_col

    #' Pull from manifest if provided
    if args.manifest and args.manifest.exists():
        man = _read_json(args.manifest)
        if preset == "none":
            preset = (man.get("dataset", {}).get("preset") or "none")
        if label is None:
            label = man.get("dataset", {}).get("target")
        if not idc:
            ids = man.get("dataset", {}).get("id_columns") or []
            idc = ids[0] if ids else None
        mrec = (man.get("preprocessing", {}).get("recipe") or {})
        if recipe == "none":
            recipe = (mrec.get("name") or "none")
        if recipe_cfg_path is None and mrec.get("config"):
            recipe_cfg_path = Path(mrec["config"])

    #' Preset-level defaults (PaySim)
    if preset == "paysim":
        label = label or PAYSIM_DEFAULT_LABEL
        idc = idc or "nameOrig"

    if not label:
        raise SystemExit("You must provide --label-col or have it in manifest/preset.")
    if not idc:
        raise SystemExit("You must provide --id-col or have it in manifest/preset.")
    if recipe not in {"none", "research-grade"}:
        raise SystemExit(f"Unknown recipe: {recipe}")
    return preset, label, idc, recipe, recipe_cfg_path

def _maybe_apply_psi(df: pd.DataFrame, args, out_dir: Path, id_col: str) -> pd.DataFrame:
    if args.psi_mode == "none":
        return df
    if not args.psi_intersection:
        raise SystemExit("--psi-intersection is required when --psi-mode != none")

    (out_dir / "psi").mkdir(parents=True, exist_ok=True)
    _write_lines(out_dir / "psi" / "used_mode.txt", [args.psi_mode])
    _write_lines(out_dir / "psi" / "intersection_tokens.txt",
                 [ln.strip() for ln in open(args.psi_intersection, "r", encoding="utf-8") if ln.strip()])

    if args.psi_mode == "hmac":
        if not args.psi_hmac_dir:
            raise SystemExit("--psi-hmac-dir is required for HMAC mode")
        canon = getattr(args, "psi_hmac_canon", "lower")
        return _filter_with_hmac(df, id_col, args.psi_hmac_dir, args.psi_intersection, canon)

    if args.psi_mode == "oprf":
        if not args.psi_client_tokens:
            raise SystemExit("--psi-client-tokens is required for OPRF mode")
        return _filter_with_oprf(df, id_col, args.psi_client_tokens, args.psi_intersection)

    return df

def main(argv: Optional[List[str]] = None) -> int:
    if pd is None:
        raise SystemExit("pandas is required for Part_2.")

    args = build_parser().parse_args(argv)
    out_dir = args.out
    _ensure_out_dir(out_dir)

    #' Align alias early
    if getattr(args, "chunk_rows", None):
        args.chunksize = args.chunk_rows

    #' Resolve defaults (preset/recipe + label id)
    preset, label_col, id_col, recipe, recipe_cfg_path = _resolve_defaults(args)
    recipe_cfg = _load_recipe_config(recipe_cfg_path) if recipe_cfg_path else None
    _log(out_dir, f"[Part_2:{__version__}] preset={preset} recipe={recipe} label={label_col} id={id_col}")

    #' Decide streaming
    input_is_csv = args.input_train.suffix.lower() not in {".parquet", ".pq"}
    do_stream = _should_stream(args, input_is_csv)
    if args.psi_mode != "none" and do_stream:
        _log(out_dir, "[stream] PSI detected → falling back to in-memory path (set --stream off to silence).")
        do_stream = False

    # ====== STREAMING PATH (CSV only, PSI=none) ======
    if do_stream and input_is_csv:
        _ = _streaming_process(
            args=args, out_dir=out_dir, preset=preset, label_col=label_col, id_col=id_col,
            step_col=args.step_col, calib_frac=args.calib_frac, test_frac=args.test_frac,
            embargo_steps=args.embargo_steps, float_dtype=args.float_dtype, chunksize=args.chunksize
        )
        if args.emit_stats:
            print((out_dir / "schema.json").read_text())
            if (out_dir / "class_weights.json").exists():
                print((out_dir / "class_weights.json").read_text())
            if (out_dir / "class_counts.json").exists():
                print((out_dir / "class_counts.json").read_text())
        return 0

    #' ====== IN-MEMORY ======
    df_train = _read_frame(args.input_train, engine=args.engine)
    df_test = _read_frame(args.input_test, engine=args.engine) if args.input_test else None

    df_train.columns = [str(c).strip() for c in df_train.columns]
    if df_test is not None:
        df_test.columns = [str(c).strip() for c in df_test.columns]

    if args.psi_mode != "none":
        df_train = _maybe_apply_psi(df_train, args, out_dir, id_col)
        if df_test is not None:
            df_test = _maybe_apply_psi(df_test, args, out_dir, id_col)

    if preset == "paysim":
        _assert_paysim_schema(df_train, label_col)
        if df_test is not None:
            _assert_paysim_schema(df_test, label_col)

    df_train = _apply_preset(df_train, preset, id_col=id_col, label_col=label_col)
    if df_test is not None:
        df_test = _apply_preset(df_test, preset, id_col=id_col, label_col=label_col)

    use_step_locked = bool(args.step_col) and (args.calib_frac > 0.0 or (args.test_frac > 0.0 and df_test is None))
    time_meta: Dict[str, Any] = {"enabled": False}
    if use_step_locked:
        df_tr_win, df_cal_win, df_te_win, time_meta = _split_windows_by_step(
            df_train, args.step_col, args.calib_frac, args.test_frac, args.embargo_steps
        )
        if df_test is not None:
            df_te_win = None
        _log(out_dir, f"[step-locked] splits -> train={len(df_tr_win)} cal={len(df_cal_win)} "
                      f"test={(len(df_te_win) if df_te_win is not None else ('ext' if df_test is not None else 0))} "
                      f"cutoffs(train_end={time_meta['cutoffs']['train_end']:.3f}, cal_end={time_meta['cutoffs']['cal_end']:.3f}) "
                      f"embargo={time_meta['embargo_steps']}")
    else:
        df_tr_win, df_cal_win, df_te_win = df_train, None, None

    schema_recipe = None
    if recipe == "research-grade":
        try:
            from preprocessing.recipes.research_grade import ResearchGradeFeaturizer
        except Exception as e:
            raise SystemExit(
                "Recipe 'research-grade' requested but module not found.\n"
                "Please add preprocessing/recipes/research_grade.py (provided later). "
                f"Import error: {e}"
            )
        fe = ResearchGradeFeaturizer(config=recipe_cfg)
        X_train, X_cal, X_test = _use_recipe_fit_transform(
            fe,
            df_tr_win,
            df_cal_win,
            (df_test if df_test is not None else df_te_win),
            id_col=id_col, label_col=label_col
        )
        schema_recipe = getattr(fe, "schema", lambda: {})() or {}

        if args.save_ids:
            _maybe_hash_ids(X_train, out_dir, True, args.hmac_keyfile)
            if X_cal is not None:
                _maybe_hash_ids(X_cal, out_dir, True, args.hmac_keyfile)
            if X_test is not None:
                _maybe_hash_ids(X_test, out_dir, True, args.hmac_keyfile)
        #' Persist artifacts (parquet bundle) to stay compatible with Part_3 / models
        _write_parquet(X_train, out_dir / "train.parquet")
        if X_cal is not None:
            _write_parquet(X_cal, out_dir / "cal.parquet")
        if X_test is not None:
            _write_parquet(X_test, out_dir / "test.parquet")

        #' STEP-locked meta CSVs (IDs + step) for audit
        if use_step_locked and args.step_col:
            _write_meta_csv(df_tr_win, id_col, args.step_col, out_dir / "meta_train.csv", "train")
            if df_cal_win is not None and len(df_cal_win):
                _write_meta_csv(df_cal_win, id_col, args.step_col, out_dir / "meta_cal.csv", "cal")
            base_test_df = (df_test if df_test is not None else df_te_win)
            if base_test_df is not None and len(base_test_df):
                _write_meta_csv(base_test_df, id_col, args.step_col, out_dir / "meta_test.csv", "test")

        #' Class weighting artifacts (same keys as baseline path)
        cw_choice = _normalize_weighting(args.weighting or args.class_weighting)
        if cw_choice == "class_balanced":
            class_weights = _balanced_weights(X_train[label_col])
            _write_json(out_dir / "class_weights.json", class_weights)
        elif cw_choice == "inverse_global":
            counts = _class_counts(X_train[label_col])
            _write_json(out_dir / "class_counts.json", counts)

        #' Persist recipe state for audit
        try:
            fe.save(out_dir / "recipe")
        except Exception:
            pass

        #' Schema  (models/centralized reads id/target)
        schema: Dict[str, Any] = {
            "version": __version__,
            "preset": preset,
            "recipe": {"name": "research-grade", "config": str(recipe_cfg_path) if recipe_cfg_path else None},
            "target": label_col,
            "id_column": id_col,
            "time_splits": time_meta,
            "recipe_schema": schema_recipe
        }
        _write_json(out_dir / "schema.json", schema)

        #' Logging
        rows_tr = len(X_train)
        rows_ca = (len(X_cal) if X_cal is not None else 0)
        rows_te = (len(X_test) if X_test is not None else 0)
        _log(out_dir, f"[Part_2:{__version__}] Done (recipe). Rows(train)={rows_tr} Rows(cal)={rows_ca} Rows(test)={rows_te} "
                      f"| recipe=research-grade | class_weighting={cw_choice}")
        if args.emit_stats:
            print((out_dir / "schema.json").read_text())
            if (out_dir / "class_weights.json").exists():
                print((out_dir / "class_weights.json").read_text())
            if (out_dir / "class_counts.json").exists():
                print((out_dir / "class_counts.json").read_text())
        return 0
    else:
        Xn_tr, Xc_tr, num_cols, cat_cols, y_tr = _split_types(df_tr_win, id_col, label_col)
        ohe_map = _fit_ohe(Xc_tr) if not Xc_tr.empty else {}
        mu, sd = _fit_scale(Xn_tr) if not Xn_tr.empty else ({}, {})

        Xc_tr_ohe = _transform_ohe(Xc_tr, ohe_map) if ohe_map else pd.DataFrame(index=Xc_tr.index)
        Xn_tr_sc = _apply_scale(Xn_tr, mu, sd) if num_cols else pd.DataFrame(index=Xn_tr.index)
        X_train = pd.concat([Xn_tr_sc, Xc_tr_ohe], axis=1)
        if args.save_ids:
            _maybe_hash_ids(df_tr_win, out_dir, True, args.hmac_keyfile)
        X_train.insert(0, id_col, df_tr_win[id_col].astype("string"))
        X_train[label_col] = y_tr.values

        X_cal = None
        if df_cal_win is not None and len(df_cal_win):
            Xn_ca, Xc_ca, _, _, y_ca = _split_types(df_cal_win, id_col, label_col)
            Xc_ca_ohe = _transform_ohe(Xc_ca, ohe_map) if ohe_map else pd.DataFrame(index=Xc_ca.index)
            for miss in set(Xc_tr_ohe.columns) - set(Xc_ca_ohe.columns):
                Xc_ca_ohe[miss] = 0
            Xc_ca_ohe = Xc_ca_ohe[Xc_tr_ohe.columns] if not Xc_tr_ohe.empty else Xc_ca_ohe
            Xn_ca_sc = _apply_scale(Xn_ca, mu, sd) if num_cols else pd.DataFrame(index=Xn_ca.index)
            X_cal = pd.concat([Xn_ca_sc, Xc_ca_ohe], axis=1)
            if args.save_ids:
                _maybe_hash_ids(df_cal_win, out_dir, True, args.hmac_keyfile)
            X_cal.insert(0, id_col, df_cal_win[id_col].astype("string"))
            X_cal[label_col] = y_ca.values

        base_test_df = (df_test if df_test is not None else df_te_win)
        X_test = None
        if base_test_df is not None and len(base_test_df):
            Xn_te, Xc_te, _, _, y_te = _split_types(base_test_df, id_col, label_col)
            Xc_te_ohe = _transform_ohe(Xc_te, ohe_map) if ohe_map else pd.DataFrame(index=Xc_te.index)
            for miss in set((Xc_tr_ohe.columns if not Xc_tr_ohe.empty else [])) - set(Xc_te_ohe.columns):
                Xc_te_ohe[miss] = 0
            if not Xc_tr_ohe.empty:
                Xc_te_ohe = Xc_te_ohe[Xc_tr_ohe.columns]
            Xn_te_sc = _apply_scale(Xn_te, mu, sd) if num_cols else pd.DataFrame(index=Xn_te.index)
            X_test = pd.concat([Xn_te_sc, Xc_te_ohe], axis=1)
            if args.save_ids:
                _maybe_hash_ids(base_test_df, out_dir, True, args.hmac_keyfile)
            X_test.insert(0, id_col, base_test_df[id_col].astype("string"))
            X_test[label_col] = y_te.values

        _write_parquet(X_train, out_dir / "train.parquet")
        if X_cal is not None:
            _write_parquet(X_cal, out_dir / "cal.parquet")
        if X_test is not None:
            _write_parquet(X_test, out_dir / "test.parquet")

        if use_step_locked and args.step_col:
            _write_meta_csv(df_tr_win, id_col, args.step_col, out_dir / "meta_train.csv", "train")
            if df_cal_win is not None and len(df_cal_win):
                _write_meta_csv(df_cal_win, id_col, args.step_col, out_dir / "meta_cal.csv", "cal")
            if df_test is None and df_te_win is not None and len(df_te_win):
                _write_meta_csv(df_te_win, id_col, args.step_col, out_dir / "meta_test.csv", "test")
            elif df_test is not None and args.step_col in df_test.columns:
                _write_meta_csv(df_test, id_col, args.step_col, out_dir / "meta_test.csv", "test")

        cw_choice = _normalize_weighting(args.weighting or args.class_weighting)
        if cw_choice == "class_balanced":
            class_weights = _balanced_weights(X_train[label_col])
            _write_json(out_dir / "class_weights.json", class_weights)
        elif cw_choice == "inverse_global":
            counts = _class_counts(X_train[label_col])
            _write_json(out_dir / "class_counts.json", counts)

        schema: Dict[str, Any] = {
            "version": __version__,
            "preset": preset,
            "recipe": {"name": recipe, "config": str(recipe_cfg_path) if recipe_cfg_path else None},
            "target": label_col,
            "id_column": id_col,
            "time_splits": time_meta
        }
        Xn_tmp, Xc_tmp, num_cols2, cat_cols2, _ = _split_types(df_tr_win, id_col, label_col)
        ohe_map_tmp = _fit_ohe(Xc_tmp) if not Xc_tmp.empty else {}
        mu_tmp, sd_tmp = _fit_scale(Xn_tmp) if not Xn_tmp.empty else ({}, {})
        schema.update({
            "numeric": num_cols2,
            "categorical": cat_cols2,
            "ohe_categories": ohe_map_tmp,
            "scaler": {"mean": mu_tmp, "std": sd_tmp}
        })
        _write_json(out_dir / "schema.json", schema)

        rows_tr = len(X_train)
        rows_ca = (len(X_cal) if X_cal is not None else 0)
        rows_te = (len(X_test) if X_test is not None else 0)
        _log(out_dir, f"[Part_2:{__version__}] Done. Rows(train)={rows_tr} Rows(cal)={rows_ca} Rows(test)={rows_te} "
                      f"| recipe={recipe} | class_weighting={cw_choice}")
        if args.emit_stats:
            print((out_dir / "schema.json").read_text())
            if (out_dir / "class_weights.json").exists():
                print((out_dir / "class_weights.json").read_text())
            if (out_dir / "class_counts.json").exists():
                print((out_dir / "class_counts.json").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
