from __future__ import annotations

import argparse, base64, hashlib, hmac, json, os, sys, time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Iterable

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:
    pa = None
    pq = None

__version__ = "0.6.2"

#' ====== GENERAL HELPERS & MANIFEST ======
def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")

def _read_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def _manifest(artifact_dir: Path) -> Dict[str, Any]:
    mpath = artifact_dir / "run_manifest.json"
    if not mpath.exists():
        raise SystemExit(f"run_manifest.json not found in {artifact_dir}. Run Part_0 first.")
    return _read_json(mpath)

def _default_id(manifest: Dict[str, Any], override: Optional[str]) -> str:
    if override: return override
    ids = manifest.get("dataset", {}).get("id_columns", [])
    if not ids:
        raise SystemExit("No id_columns in manifest; pass --id-col explicitly.")
    return ids[0]

def _ensure_party_dir(base: Path, party: str) -> Path:
    d = base / party
    d.mkdir(parents=True, exist_ok=True)
    return d

def _detect_engine(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".parquet", ".pq"): return "parquet"
    if ext in (".csv", ".txt"):     return "csv"
    raise SystemExit(f"Unsupported input extension: {ext}")

def _iter_rows(path: Path, engine: str, chunksize: int, columns: List[str]) -> Iterable[pd.DataFrame]:
    if engine == "csv":
        for chunk in pd.read_csv(path, chunksize=chunksize, usecols=columns, dtype=str, low_memory=False):
            yield chunk
    else:
        df = pd.read_parquet(path, columns=columns)
        if len(df) <= chunksize:
            yield df
        else:
            for i in range(0, len(df), chunksize):
                yield df.iloc[i:i+chunksize].copy()

def _canonize(s: Optional[str], mode: str) -> str:
    """
    Canonicalize ID strings in a deterministic way so PSI tokens match across parties:
      - 'none'  : strip only
      - 'lower' : strip + lowercase
      - 'upper' : strip + uppercase
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        s = ""
    s = str(s).strip()
    if mode == "lower":
        return s.lower()
    if mode == "upper":
        return s.upper()
    return s


# ====== HMAC PSI ======
def _key_path(base: Path) -> Path:
    return base / "hmac_key.json"

def _load_or_create_hmac_key(base: Path, rotate: bool = False) -> Dict[str, str]:
    p = _key_path(base)
    if p.exists() and not rotate:
        return _read_json(p)
    k = os.urandom(32)  # 256-bit
    salt = os.urandom(16)
    obj = {"k": base64.b64encode(k).decode(), "salt": base64.b64encode(salt).decode()}
    _write_json(p, obj)
    return obj

def _hmac_tokenize(ids: Iterable[str], key: bytes, salt: bytes, *, canon: str = "none") -> List[str]:
    out = []
    for s in ids:
        msg = salt + _canonize(s, canon).encode("utf-8", errors="ignore")
        t = hmac.new(key, msg, hashlib.sha256).hexdigest()
        out.append(t)
    return out

def cmd_hmac_init(args: argparse.Namespace) -> None:
    psi_root = Path(args.artifact_dir)
    psi_root.mkdir(parents=True, exist_ok=True)
    _write_json(psi_root / "manifest_link.json",
                {"run_manifest": "./run_manifest.json", "created_at": _now(), "mode": "hmac"})
    _load_or_create_hmac_key(psi_root, rotate=args.rotate)
    print(f"[HMAC] key @ {psi_root/'hmac_key.json'}  (rotate={args.rotate})")

#' PSI rootes
def cmd_hmac_generate(args: argparse.Namespace) -> None:
    psi_root = Path(args.artifact_dir)
    man = _manifest(psi_root)
    party_dir = _ensure_party_dir(psi_root, args.party)
    keyobj = _load_or_create_hmac_key(psi_root, rotate=False)

    id_col = _default_id(man, args.id_col)
    path = Path(args.input); eng = _detect_engine(path)
    chunksize = int(args.chunksize)

    k = base64.b64decode(keyobj["k"]); salt = base64.b64decode(keyobj["salt"])

    n_rows = 0
    tokens_txt = party_dir / "tokens.txt"
    tokens_pq  = party_dir / "tokens.parquet"

    #' init outputs
    if tokens_txt.exists(): tokens_txt.unlink()
    tok_accum: List[str] = []

    #' Maap writer (chunk-safe)
    map_writer = None
    map_target_parquet = party_dir / "id_token_map.parquet"
    map_target_csv = party_dir / "id_token_map.csv"  # fallback from pyarrow if issue

    for chunk in _iter_rows(path, eng, chunksize, [id_col]):
        ids = chunk[id_col].astype(str).tolist()
        toks = _hmac_tokenize(ids, k, salt, canon=args.canon)
        tok_accum.extend(toks)
        n_rows += len(chunk)

        #' flush periodically to .txt
        if len(tok_accum) >= 100_000:
            with open(tokens_txt, "a", encoding="utf-8") as f:
                f.write("\n".join(tok_accum) + "\n")
            tok_accum = []

        #' optional map
        if args.emit_map:
            dfm = pd.DataFrame({id_col: ids, "token": toks})
            if pa is not None and pq is not None:
                table = pa.Table.from_pandas(dfm, preserve_index=False)
                if map_writer is None:
                    map_writer = pq.ParquetWriter(map_target_parquet, table.schema, compression="zstd")
                map_writer.write_table(table)
            else:
                header = not map_target_csv.exists()
                dfm.to_csv(map_target_csv, index=False, mode="a", header=header)

    #' final flush for tokens.txt
    if tok_accum:
        with open(tokens_txt, "a", encoding="utf-8") as f:
            f.write("\n".join(tok_accum) + "\n")

    #' close parquet writer if used
    if map_writer is not None:
        map_writer.close()

    #'writes tokens.parquet once if pyarrow is available; else write CSV fallback
    try:
        with open(tokens_txt, "r", encoding="utf-8") as f:
            toks_all = [line.strip() for line in f if line.strip()]
        if pa is not None and pq is not None:
            pd.DataFrame({"token": toks_all}).to_parquet(tokens_pq, index=False)
        else:
            pd.DataFrame({"token": toks_all}).to_csv(party_dir / "tokens.csv", index=False)
    except Exception as e:
        print(f"[HMAC] Note: could not create tokens.parquet (ok to ignore): {e}", file=sys.stderr)

    _write_json(party_dir / "log.json", {
        "mode": "hmac", "input": str(path), "id_col": id_col, "rows": n_rows,
        "chunksize": chunksize, "canon": args.canon, "created_at": _now(), "version": __version__
    })

    print(f"[HMAC] {n_rows} rows → {tokens_txt}")


# ====== OPRF PSI ======
class _OPRFAdapter:
    """
    Minimal adapter that tries `voprf` first, then `oprf`.
    Exposes:
      - setup_server(suite) -> bytes (serialized sk)
      - blind_ids(ids, suite) -> (blinded_bytes, client_state_bytes)
      - evaluate_server(sk_bytes, blinded_bytes, suite) -> bytes
      - finalize_client(client_state_bytes, evaluated_bytes, suite) -> List[str] tokens(hex)
    If libs are missing, raises with installation guidance.
    """
    def __init__(self):
        self.mode = None
        try:
            import voprf
            self.mode = "voprf"
            self.v = voprf
        except Exception:
            try:
                import oprf
                self.mode = "oprf"
                self.v = oprf
            except Exception:
                self.mode = None

    def _ensure(self):
        if not self.mode:
            raise RuntimeError("No OPRF library found. Install one of:\n"
                               "  pip install voprf   # or\n"
                               "  pip install oprf    # RFC 9497 compatible")

    def setup_server(self, suite: str) -> bytes:
        self._ensure()
        try:
            ctx = self.v.Server(self._suite(suite))
            return ctx.export_key()
        except Exception as e:
            raise RuntimeError(f"{self.mode} setup failed: {e}")

    def blind_ids(self, ids: List[str], suite: str) -> Tuple[bytes, bytes]:
        self._ensure()
        try:
            client = self.v.Client(self._suite(suite))
            blinds, st = [], []
            for s in ids:
                b, state = client.blind(str(s).encode("utf-8"))
                blinds.append(b); st.append(state)
            return self.v.serialize_list(blinds), self.v.serialize_list(st)
        except Exception as e:
            raise RuntimeError(f"{self.mode} blind failed: {e}")

    def evaluate_server(self, sk_bytes: bytes, blinded_bytes: bytes, suite: str) -> bytes:
        self._ensure()
        try:
            server = self.v.Server(self._suite(suite), key=sk_bytes)
            blinds = self.v.deserialize_list(blinded_bytes)
            evals = [server.evaluate(b) for b in blinds]
            return self.v.serialize_list(evals)
        except Exception as e:
            raise RuntimeError(f"{self.mode} evaluate failed: {e}")

    def finalize_client(self, client_state_bytes: bytes, evaluated_bytes: bytes, suite: str) -> List[str]:
        self._ensure()
        try:
            client = self.v.Client(self._suite(suite))
            states = self.v.deserialize_list(client_state_bytes)
            evals  = self.v.deserialize_list(evaluated_bytes)
            toks = [client.finalize(st, ev).hex() for st, ev in zip(states, evals)]
            return toks
        except Exception as e:
            raise RuntimeError(f"{self.mode} finalize failed: {e}")

    def _suite(self, name: str):
        return name


def cmd_oprf_setup(args: argparse.Namespace) -> None:
    psi_root = Path(args.artifact_dir)
    psi_root.mkdir(parents=True, exist_ok=True)
    _write_json(psi_root / "manifest_link.json",
                {"run_manifest": "./run_manifest.json", "created_at": _now(), "mode": "oprf"})

    adapter = _OPRFAdapter()
    sk = adapter.setup_server(args.suite)
    with open(psi_root / "oprf_server_sk.bin", "wb") as f:
        f.write(sk)
    print(f"[OPRF] Server key created @ {psi_root/'oprf_server_sk.bin'} (suite={args.suite})")

def cmd_oprf_blind(args: argparse.Namespace) -> None:
    psi_root = Path(args.artifact_dir); man = _manifest(psi_root)
    party_dir = _ensure_party_dir(psi_root, args.party)

    id_col = _default_id(man, args.id_col)
    path = Path(args.input); eng = _detect_engine(path)
    chunksize = int(args.chunksize)

    adapter = _OPRFAdapter()
    out_blinded = party_dir / "blinded.bin"
    out_state   = party_dir / "client_state.bin"

    all_ids: List[str] = []
    for chunk in _iter_rows(path, eng, chunksize, [id_col]):
        ids = chunk[id_col].astype(str).tolist()
        all_ids.extend([_canonize(v, args.canon) for v in ids])

    blinded, state = adapter.blind_ids(all_ids, args.suite)
    with open(out_blinded, "wb") as f: f.write(blinded)
    with open(out_state, "wb") as f:   f.write(state)

    _write_json(party_dir / "log.json", {
        "mode": "oprf", "step": "blind", "rows": len(all_ids), "id_col": id_col,
        "input": str(path), "suite": args.suite, "canon": args.canon,
        "created_at": _now(), "version": __version__
    })
    print(f"[OPRF] Blinded {len(all_ids)} IDs → {out_blinded}")

def cmd_oprf_evaluate(args: argparse.Namespace) -> None:
    psi_root = Path(args.artifact_dir)
    sk = (psi_root / "oprf_server_sk.bin").read_bytes()
    blinded = Path(args.blinded).read_bytes()

    adapter = _OPRFAdapter()
    evaluated = adapter.evaluate_server(sk, blinded, args.suite)
    out = Path(args.out) if args.out else (Path(args.blinded).with_suffix(".evaluated.bin"))
    with open(out, "wb") as f: f.write(evaluated)
    print(f"[OPRF] Evaluated → {out}")

def cmd_oprf_finalize(args: argparse.Namespace) -> None:
    psi_root = Path(args.artifact_dir)
    party_dir = _ensure_party_dir(psi_root, args.party)

    adapter = _OPRFAdapter()
    state = (party_dir / "client_state.bin").read_bytes()
    evaluated = Path(args.evaluated).read_bytes()

    tokens = adapter.finalize_client(state, evaluated, args.suite)
    #' write tokens
    tokens_txt = party_dir / "tokens.txt"
    tokens_pq  = party_dir / "tokens.parquet"
    with open(tokens_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(tokens) + ("\n" if tokens else ""))
    if pa is not None and pq is not None:
        pd.DataFrame({"token": tokens}).to_parquet(tokens_pq, index=False)
    else:
        pd.DataFrame({"token": tokens}).to_csv(party_dir / "tokens.csv", index=False)

    _write_json(party_dir / "log.json", {
        "mode": "oprf", "step": "finalize", "rows": len(tokens),
        "created_at": _now(), "version": __version__
    })
    print(f"[OPRF] Finalized {len(tokens)} tokens → {tokens_txt}")


# ====== INTERSECTIONS ======
def _load_tokens_any(p: Path) -> List[str]:
    """Load tokens from .parquet/.csv/.txt, returning a list of strings."""
    ext = p.suffix.lower()
    if ext in (".parquet", ".pq"):
        df = pd.read_parquet(p)
        col = "token" if "token" in df.columns else df.columns[0]
        return df[col].astype(str).str.strip().tolist()
    if ext == ".csv":
        df = pd.read_csv(p)
        col = "token" if "token" in df.columns else df.columns[0]
        return df[col].astype(str).str.strip().tolist()
    with open(p, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def cmd_intersect(args: argparse.Namespace) -> None:
    psi_root = Path(args.artifact_dir)
    out_dir = psi_root / "intersections"
    out_dir.mkdir(parents=True, exist_ok=True)

    token_sets = []
    counts = {}
    for p in args.tokens:
        toks = set(_load_tokens_any(Path(p)))
        token_sets.append(toks)
        counts[Path(p).parent.name] = len(toks)

    if not token_sets:
        raise SystemExit("No token files given.")
    inter = set.intersection(*token_sets)

    out_txt = out_dir / f"{args.label}_intersection.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(inter)) + ("\n" if inter else ""))

    _write_json(out_dir / f"{args.label}_counts.json", {
        "parties": counts, "intersection": len(inter), "created_at": _now()
    })

    #' flat path for runner compatibility:
    legacy = psi_root / "intersection.txt"
    with open(legacy, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(inter)) + ("\n" if inter else ""))

    print(f"[PSI] Intersection | parties={len(token_sets)}  "
          f"min={min(map(len, token_sets))} max={max(map(len, token_sets))}  "
          f"∩={len(inter)} → {out_txt}  (legacy: {legacy})")


# ====== CLI ======
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="Part_1A_psi",
        description="Private Set Intersection for ID alignment (HMAC or OPRF)."
    )
    p.add_argument("--artifact-dir", default="artifacts/run/psi",
                   help="PSI root dir. Your runner passes $OUT/psi here; we write inside it.")

    sp = p.add_subparsers(dest="cmd", required=True)

    #' HMAC init
    s = sp.add_parser("psi-hmac-init", help="Create/rotate HMAC key+salt in <artifact-dir>/hmac_key.json.")
    s.add_argument("--rotate", action="store_true", help="Rotate key even if exists.")
    s.set_defaults(func=cmd_hmac_init)

    #' HMAC generate
    s = sp.add_parser("psi-hmac-generate", help="Tokenize IDs with HMAC-SHA256.")
    s.add_argument("--party", required=True, help="Party tag for output subdir (e.g., client_1).")
    s.add_argument("--input", required=True, help="CSV or Parquet file with ID column.")
    s.add_argument("--id-col", default=None, help="ID column (default: first id_columns from manifest).")
    s.add_argument("--chunksize", type=int, default=200_000)
    s.add_argument("--canon", choices=["none","lower","upper"], default="none",
                   help="Canonicalize IDs before tokenization (must match across parties).")

    s.add_argument("--emit-map", action="store_true", help="Also write id_token_map.parquet (PII!).")
    s.set_defaults(func=cmd_hmac_generate)

    #' OPRF setup (server)
    s = sp.add_parser("psi-oprf-setup", help="Server: initialize OPRF secret key.")
    s.add_argument("--suite", default="ristretto255", help="OPRF suite (library-dependent, e.g., ristretto255 or P256_SHA256).")
    s.set_defaults(func=cmd_oprf_setup)

    #' OPRF blind (client)
    s = sp.add_parser("psi-oprf-blind", help="Client: blind local IDs.")
    s.add_argument("--party", required=True)
    s.add_argument("--input", required=True)
    s.add_argument("--id-col", default=None)
    s.add_argument("--suite", default="ristretto255")
    s.add_argument("--chunksize", type=int, default=500_000)
    s.add_argument("--canon", choices=["none","lower","upper"], default="none",
                   help="Canonicalize IDs before blinding (must match across parties).")

    s.set_defaults(func=cmd_oprf_blind)

    #' OPRF evaluate (server)
    s = sp.add_parser("psi-oprf-evaluate", help="Server: evaluate blinded inputs.")
    s.add_argument("--blinded", required=True, help="Path to <party>/blinded.bin from client.")
    s.add_argument("--out", default=None, help="Optional output filename; default is <blinded>.evaluated.bin")
    s.add_argument("--suite", default="ristretto255")
    s.set_defaults(func=cmd_oprf_evaluate)

    #' OPRF finalize (client)
    s = sp.add_parser("psi-oprf-finalize", help="Client: finalize evaluated blinds into tokens.")
    s.add_argument("--party", required=True)
    s.add_argument("--evaluated", required=True, help="Path to evaluated.bin from server.")
    s.add_argument("--suite", default="ristretto255")
    s.set_defaults(func=cmd_oprf_finalize)

    #' Intersections (generic)
    s = sp.add_parser("psi-intersect", help="Compute intersection across token files.")
    s.add_argument("--label", required=True, help="Label for outputs (e.g., bankA_vs_bankB).")
    s.add_argument("--tokens", nargs="+", required=True, help="Paths to tokens files (.parquet/.csv/.txt) from each party.")
    s.set_defaults(func=cmd_intersect)

    return p

def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
