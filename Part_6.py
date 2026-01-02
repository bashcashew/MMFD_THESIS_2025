from __future__ import annotations

import argparse, base64, hashlib, hmac, json, os, sys, time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np

__version__ = "0.7.0"

#' ---------------------------------------------------------------------
#' Small helpers
#' ---------------------------------------------------------------------
def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")

def _write_json(p: Path, o: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(o, f, indent=2, sort_keys=True)

def _read_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _hmac_drbg(key: bytes, salt: bytes, counter: int) -> bytes:
    """HMAC-SHA256(key, salt || counter_be) → 32 bytes."""
    ctr = counter.to_bytes(8, "big")
    return hmac.new(key, salt + ctr, hashlib.sha256).digest()

def _prg_float_vec(key: bytes, salt: bytes, dim: int) -> np.ndarray:
    """Deterministically expand key into dim float64 values in (-0.5, 0.5)."""
    out = np.empty(dim, dtype=np.float64)
    #' 8 bytes per float (via uint64); 32-byte block -> 4 floats
    blocks = (dim + 3) // 4
    idx = 0
    for c in range(blocks):
        block = _hmac_drbg(key, salt, c)
        u64 = np.frombuffer(block, dtype=np.uint64, count=4)
        #' map to [0,1) via /2**64 then center to (-0.5,0.5)
        vals = (u64.astype(np.float64) / 18446744073709551616.0) - 0.5
        take = min(4, dim - idx)
        out[idx:idx+take] = vals[:take]
        idx += take
    return out

def _session_dir(artifact_dir: Path, label: str) -> Path:
    return artifact_dir / "secureagg" / label

def _client_dir(artifact_dir: Path, label: str, client_id: str) -> Path:
    return _session_dir(artifact_dir, label) / "clients" / client_id

#' ---------------------------------------------------------------------
#' CMDs
#' ---------------------------------------------------------------------
def cmd_session_init(args: argparse.Namespace) -> None:
    sd = _session_dir(Path(args.artifact_dir), args.label)
    _ensure_dir(sd / "clients")
    salt = os.urandom(16)  #' 128-bit session salt
    clients = [c.strip() for c in args.clients.split(",") if c.strip()]
    clients_sorted = sorted(set(clients))
    if len(clients_sorted) < 2:
        raise SystemExit("Need at least 2 clients for secure aggregation.")

    session = {
        "version": __version__,
        "label": args.label,
        "dim": int(args.dim),
        "mode": str(args.mode),
        "scale": float(args.scale) if args.mode == "fixed" else None,
        "salt_hex": base64.b16encode(salt).decode("ascii"),
        "clients": clients_sorted,
        "created_at": _now()
    }
    _write_json(sd / "session.json", session)
    print(f"[secureagg] session-init → {sd/'session.json'}  clients={len(clients_sorted)} dim={args.dim}")

def _dev_generate_pairwise(clients: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Create symmetric 32B secrets for each unordered pair.
    Output: {A: {B: base64key, ...}, B: {A: base64key, ...}, ...}
    """
    import secrets
    m: Dict[str, Dict[str, str]] = {c: {} for c in clients}
    for i, a in enumerate(clients):
        for b in clients[i+1:]:
            k = secrets.token_bytes(32)
            k_b64 = base64.b64encode(k).decode("ascii")
            m[a][b] = k_b64
            m[b][a] = k_b64
    return m

def cmd_dev_gensecrets(args: argparse.Namespace) -> None:
    sd = _session_dir(Path(args.artifact_dir), args.label)
    session = _read_json(sd / "session.json")
    clients = session["clients"]
    out = _dev_generate_pairwise(clients)
    dev_dir = sd / "dev"
    _ensure_dir(dev_dir)
    _write_json(dev_dir / "pairwise_secrets.json", out)
    print(f"[secureagg] dev-gensecrets → {dev_dir/'pairwise_secrets.json'}")

def cmd_client_derive(args: argparse.Namespace) -> None:
    art = Path(args.artifact_dir); label = args.label; cid = args.client_id
    sd = _session_dir(art, label); cd = _client_dir(art, label, cid)
    _ensure_dir(cd)
    sess = _read_json(sd / "session.json")
    try:
        clients = sess["clients"]
        dim = int(sess["dim"]); mode = sess["mode"]; scale = sess.get("scale")
        salt = base64.b16decode(sess["salt_hex"].encode("ascii"))
    except Exception as e:
        raise SystemExit(f"Bad session.json: {e}")

    if cid not in clients:
        raise SystemExit(f"client-id {cid} not in session client list.")

    #' Load pairwise secrets
    secrets_map = _read_json(Path(args.pair_secrets))
    peers = [p for p in clients if p != cid]
    for p in peers:
        if p not in secrets_map.get(cid, {}):
            raise SystemExit(f"Missing secret for peer '{p}' under client '{cid}' in {args.pair_secrets}")

    #' Build combined mask
    mask = np.zeros(dim, dtype=np.float64)
    for peer in peers:
        k_b64 = secrets_map[cid][peer]
        key = base64.b64decode(k_b64.encode("ascii"))
        pair_salt = hashlib.sha256(salt + b"|" + min(cid, peer).encode() + b"|" + max(cid, peer).encode()).digest()
        m = _prg_float_vec(key, pair_salt, dim)  #' float in (-0.5, 0.5)
        #' Signage: lower-lex client adds, higher-lex subtracts
        if cid < peer:
            mask += m
        else:
            mask -= m


    if mode == "fixed":
        s = float(scale or 1.0)
        mask = np.round(mask * s).astype(np.float64)


    mask_path = cd / "mask.npy"
    np.save(mask_path, mask)

    commit_key = hashlib.sha256(b"ffd_secureagg_commit|" + salt + cid.encode()).digest()
    commit = hmac.new(commit_key, mask.tobytes(), hashlib.sha256).hexdigest()
    _write_json(cd / "commit.json", {
        "client_id": cid, "label": label, "dim": dim, "mode": mode,
        "salt_hex": sess["salt_hex"], "commit": commit, "created_at": _now()
    })
    print(f"[secureagg] client-derive → {mask_path}  (||mask||={float(np.linalg.norm(mask)):.6g})")

def _load_vec(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    #' JSON lis
    try:
        arr = np.asarray(_read_json(path), dtype=np.float64)
        return arr
    except Exception:
        raise SystemExit(f"Unsupported vector file: {path} (use .npy or JSON list)")

def cmd_client_apply(args: argparse.Namespace) -> None:
    art = Path(args.artifact_dir); label = args.label; cid = args.client_id
    cd = _client_dir(art, label, cid)
    mask = np.load(cd / "mask.npy")
    g = _load_vec(Path(args.grad_sum))
    if g.shape != mask.shape:
        raise SystemExit(f"grad_sum dim {g.shape} != mask dim {mask.shape}")
    masked = g + mask
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.save(outp, masked)
    #' Also copy into client folder if output points elsewhere
    (cd / "masked.npy").write_bytes(outp.read_bytes())
    print(f"[secureagg] client-apply → {outp}  (masked)")

def cmd_server_sum(args: argparse.Namespace) -> None:
    vecs = [np.load(Path(p)) for p in args.masked]
    if not vecs:
        raise SystemExit("Provide at least one --masked file.")
    dim = vecs[0].shape
    agg = np.zeros(dim, dtype=np.float64)
    for v in vecs:
        if v.shape != dim:
            raise SystemExit("All masked vectors must share the same dimension.")
        agg += v
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.save(outp, agg)
    print(f"[secureagg] server-sum → {outp} (shape={agg.shape})")

#' HE passthrough sanity
def cmd_he_check(args: argparse.Namespace) -> None:
    scheme = args.scheme
    he_dir = Path(args.he_dir) if args.he_dir else None
    x = _load_vec(Path(args.vector))
    if scheme == "paillier":
        from crypto.he_paillier import PaillierContext
        ctx = PaillierContext.load_public(he_dir)
        ct = ctx.encrypt_numpy_to_bytes(x)
        print(f"[he-check] paillier encrypted bytes={len(ct)}")
    elif scheme == "ckks":
        from crypto.he_ckks import CKKSContext
        ctx = CKKSContext.load_public(he_dir)
        ct = ctx.encrypt_numpy_to_bytes(x)
        print(f"[he-check] ckks encrypted bytes={len(ct)}")
    else:
        print("[he-check] scheme=none (no action)")

#' ---------------------------------------------------------------------
#' CLI
#' ---------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Part_6: Secure aggregation toolbox (pairwise masks + HE passthrough)."
    )
    p.add_argument("--artifact-dir", default="artifacts/run")

    sp = p.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("session-init", help="Create a new secure-aggregation session descriptor.")
    s.add_argument("--label", required=True, help="Session label (e.g. 'round1').")
    s.add_argument("--dim", type=int, required=True, help="Vector dimension (LR θ dimension).")
    s.add_argument("--mode", choices=["float","fixed"], default="float", help="Float64 masks or fixed-point scaled.")
    s.add_argument("--scale", type=float, default=1e6, help="Scale for fixed-point mode.")
    s.add_argument("--clients", required=True, help="Comma-separated client IDs.")
    s.set_defaults(func=cmd_session_init)

    s = sp.add_parser("dev-gensecrets", help="DEV ONLY: generate pairwise 32B secrets for this session.")
    s.add_argument("--label", required=True)
    s.set_defaults(func=cmd_dev_gensecrets)

    s = sp.add_parser("client-derive", help="Derive and store the local mask for a client.")
    s.add_argument("--label", required=True)
    s.add_argument("--client-id", required=True)
    s.add_argument("--pair-secrets", required=True, help="JSON mapping peers to base64 keys; dev output is accepted.")
    s.set_defaults(func=cmd_client_derive)

    s = sp.add_parser("client-apply", help="Apply stored mask to a local gradient-sum vector (JSON or .npy).")
    s.add_argument("--label", required=True)
    s.add_argument("--client-id", required=True)
    s.add_argument("--grad-sum", required=True, help="Path to local gradient-sum vector (.npy or JSON list).")
    s.add_argument("--out", required=True, help="Output .npy for masked vector.")
    s.set_defaults(func=cmd_client_apply)

    s = sp.add_parser("server-sum", help="Sum masked vectors to recover plaintext aggregate (demo/helper).")
    s.add_argument("--label", required=True)
    s.add_argument("--masked", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_server_sum)

    s = sp.add_parser("he-check", help="(Optional) Encrypt a vector with HE loaders for sanity.")
    s.add_argument("--scheme", choices=["none","paillier","ckks"], default="none")
    s.add_argument("--he-dir", default=None)
    s.add_argument("--vector", required=True)
    s.set_defaults(func=cmd_he_check)

    return p

def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
