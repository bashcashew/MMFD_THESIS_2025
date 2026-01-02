from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

__version__ = "0.6.0"


from utils.io import write_json as _io_write_json, atomic_write_bytes as _io_write_bytes


def _try_import_phe():
    try:
        import phe
        from phe import paillier
        return phe, paillier
    except Exception as e:
        raise SystemExit(
            "Paillier requires the 'phe' package.\n"
            "  pip install phe\n"
            f"Import error: {e}"
        )

def _try_import_tenseal():
    try:
        import tenseal as ts
        return ts
    except Exception as e:
        raise SystemExit(
            "CKKS requires the 'tenseal' package (and CPU support).\n"
            "  pip install tenseal\n"
            f"Import error: {e}"
        )

#' ====== UTILITIESS ======
def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")

def _read_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    _io_write_json(p, obj, sort_keys=True, compact=False)

def _write_bytes(p: Path, b: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    _io_write_bytes(p, b)

def _manifest(artifact_dir: Path) -> Dict[str, Any]:
    mpath = artifact_dir / "run_manifest.json"
    if not mpath.exists():
        raise SystemExit(f"[he] run_manifest.json not found in {artifact_dir}. Run Part_0 first.")
    return _read_json(mpath)

def _link_manifest(he_dir: Path) -> None:
    _write_json(he_dir / "manifest_link.json", {
        "run_manifest": "../../run_manifest.json",
        "created_at": _now(),
        "version": __version__,
    })

def _best_effort_zeroize(path: Path) -> None:
    """
    Try to securely wipe a file; if optional helper exists, use it; else overwrite+unlink.
    """
    try:
        #' Optional repo module (crypto/zeroize.py)
        from crypto.zeroize import best_effort_wipe
        best_effort_wipe(path)
        return
    except Exception:
        pass
    try:
        if path.exists() and path.is_file():
            sz = path.stat().st_size
            with open(path, "r+b") as f:
                f.seek(0)
                f.write(b"\x00" * sz)
                f.flush()
                os.fsync(f.fileno())
            path.unlink(missing_ok=True)
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

# ====== PAILLIER CMDs ======
def cmd_paillier_init(args: argparse.Namespace) -> None:
    _, paillier = _try_import_phe()
    art = Path(args.artifact_dir)
    he_dir = art / "he"
    p_dir = he_dir / "paillier"
    p_dir.mkdir(parents=True, exist_ok=True)
    _link_manifest(he_dir)

    #' rotate handling
    if args.rotate:
        for f in [p_dir / "private.json", p_dir / "public.json"]:
            if f.exists():
                _best_effort_zeroize(f)
        print("[he/paillier] rotated keys: previous files zeroized (if present).", file=sys.stderr)

    key_bits = max(1024, int(args.key_bits))
    if key_bits < 2048:
        print(f"[he/paillier] WARNING: {key_bits}-bit modulus is weak; prefer >= 2048.", file=sys.stderr)

    pub, priv = paillier.generate_paillier_keypair(n_length=key_bits)

    #' store decimal strings (no pickles; this can be "unsafe")
    pub_j = {"n": str(pub.n)}
    priv_j = {"p": str(priv.p), "q": str(priv.q)}

    _write_json(p_dir / "public.json", pub_j)
    _write_json(p_dir / "private.json", priv_j)

    print(f"[he/paillier] generated {key_bits}-bit keypair → {p_dir}")
    print("  public.json (share with clients), private.json (server-only)")

def cmd_paillier_export(args: argparse.Namespace) -> None:
    art = Path(args.artifact_dir); dest = Path(args.dest)
    src = art / "he" / "paillier" / "public.json"
    if not src.exists():
        raise SystemExit("paillier/public.json not found. Run he-paillier-init first.")
    dest.mkdir(parents=True, exist_ok=True)
    data = _read_json(src)
    _write_json(dest / "public.json", data)
    print(f"[he/paillier] exported public key → {dest/'public.json'}")

def cmd_paillier_selftest(args: argparse.Namespace) -> None:
    _, paillier = _try_import_phe()
    art = Path(args.artifact_dir)
    p_dir = art / "he" / "paillier"
    pub_j = _read_json(p_dir / "public.json")
    priv_j = _read_json(p_dir / "private.json")

    pub = paillier.PaillierPublicKey(n=int(pub_j["n"]))
    priv = paillier.PaillierPrivateKey(public_key=pub, p=int(priv_j["p"]), q=int(priv_j["q"]))

    #' integer test: encrypt [1,2,3], add ciphertexts, decrypt sum==6
    xs = [1, 2, 3]
    cts = [pub.encrypt(x) for x in xs]
    csum = cts[0]
    for c in cts[1:]:
        csum = csum + c
    s = priv.decrypt(csum)
    ok = (s == sum(xs))
    print(f"[he/paillier] selftest: sum({xs}) → decrypt({s})  OK={ok}")
    if not ok:
        raise SystemExit("Paillier self-test failed.")

#' ====== CKKS CMDs ======
def _parse_bits_list(s: str) -> List[int]:
    #'supports "60,40,40,60" or "60 40 40 60"
    parts = [p for p in str(s).replace(",", " ").split() if p]
    try:
        return [int(p) for p in parts]
    except Exception:
        raise SystemExit(f"Invalid --coeff-mod-bits: {s}")

def _parse_scale(s: str) -> float:
    #' supports "2**40" or numeric float
    st = str(s).strip()
    if st.startswith("2**"):
        n = int(st[3:])
        return float(2 ** n)
    try:
        return float(st)
    except Exception:
        raise SystemExit(f"Invalid --scale: {s}")

def cmd_ckks_init(args: argparse.Namespace) -> None:
    ts = _try_import_tenseal()
    art = Path(args.artifact_dir)
    he_dir = art / "he"
    c_dir = he_dir / "ckks"
    c_dir.mkdir(parents=True, exist_ok=True)
    _link_manifest(he_dir)

    #' rotate handling (zeroizes the previous files)
    if args.rotate:
        for f in [c_dir / "context_server.bin", c_dir / "context_clients.bin", c_dir / "meta.json"]:
            if f.exists():
                _best_effort_zeroize(f)
        print("[he/ckks] rotated contexts: previous files zeroized (if present).", file=sys.stderr)

    poly = int(args.poly_mod_degree)
    coeffs = _parse_bits_list(args.coeff_mod_bits)
    scale = _parse_scale(args.scale)

    if poly < 8192:
        print(f"[he/ckks] WARNING: poly_mod_degree={poly} is small for many workloads; 8192+ recommended.", file=sys.stderr)

    #' build context
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly,
        coeff_mod_bit_sizes=coeffs
    )
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    ctx.global_scale = scale

    #' serialize two variants
    ctx_ser_pub = ctx.serialize(save_secret_key=False)
    ctx_ser_sec = ctx.serialize(save_secret_key=True)

    _write_bytes(c_dir / "context_clients.bin", ctx_ser_pub)
    _write_bytes(c_dir / "context_server.bin", ctx_ser_sec)
    _write_json(c_dir / "meta.json", {
        "poly_modulus_degree": poly,
        "coeff_mod_bit_sizes": coeffs,
        "scale": scale,
        "created_at": _now(),
        "version": __version__
    })

    print(f"[he/ckks] context created (poly={poly}, coeffs={coeffs}, scale={scale:g}) → {c_dir}")
    print("  context_clients.bin (share with clients), context_server.bin (server-only)")

def cmd_ckks_export(args: argparse.Namespace) -> None:
    art = Path(args.artifact_dir); dest = Path(args.dest)
    src_ctx = art / "he" / "ckks" / "context_clients.bin"
    src_meta = art / "he" / "ckks" / "meta.json"
    if not src_ctx.exists() or not src_meta.exists():
        raise SystemExit("ckks context not found. Run he-ckks-init first.")
    dest.mkdir(parents=True, exist_ok=True)
    _write_bytes(dest / "context_clients.bin", src_ctx.read_bytes())
    _write_json(dest / "meta.json", _read_json(src_meta))
    print(f"[he/ckks] exported public context → {dest/'context_clients.bin'} (+meta.json)")

def cmd_ckks_selftest(args: argparse.Namespace) -> None:
    ts = _try_import_tenseal()
    art = Path(args.artifact_dir)
    c_dir = art / "he" / "ckks"
    pub_path = c_dir / "context_clients.bin"
    sec_path = c_dir / "context_server.bin"
    if not pub_path.exists() or not sec_path.exists():
        raise SystemExit("CKKS contexts not found. Run he-ckks-init first.")

    pub_bin = pub_path.read_bytes()
    sec_bin = sec_path.read_bytes()

    #' client: encrypt two vectors with public context
    ctx_pub = ts.context_from(pub_bin)
    v1 = [0.1, 0.2, 0.3, 0.4]
    v2 = [1.0, 2.0, 3.0, 4.0]
    c1 = ts.ckks_vector(ctx_pub, v1)
    c2 = ts.ckks_vector(ctx_pub, v2)
    csum = c1 + c2

    #' server: decrypt sum with secret context
    ctx_sec = ts.context_from(sec_bin)
    s = csum.decrypt(ctx_sec)

    #' verify CKKS approximate correctness (tolerance)
    target = [a + b for a, b in zip(v1, v2)]
    ok = all(abs(a - b) < 1e-6 for a, b in zip(s, target))
    print(f"[he/ckks] selftest: decrypt({s}) ≈ {target}  OK={ok}")
    if not ok:
        raise SystemExit("CKKS self-test failed (tolerance exceeded).")

# ====== CLI ======
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="Part_1B_he_keys",
        description="Generate and distribute Paillier/CKKS HE materials."
    )
    p.add_argument("--artifact-dir", default="artifacts/run",
                   help="Directory with run_manifest.json and where he/ is written.")

    sp = p.add_subparsers(dest="cmd", required=True)

    #' Paillier
    s = sp.add_parser("he-paillier-init", help="Generate Paillier keypair.")
    s.add_argument("--key-bits", type=int, default=2048, help="Modulus bits (>= 2048 recommended).")
    s.add_argument("--rotate", action="store_true", help="Rotate keys (zeroize old files if present).")
    s.set_defaults(func=cmd_paillier_init)

    s = sp.add_parser("he-paillier-export", help="Export Paillier public key for clients.")
    s.add_argument("--dest", required=True, help="Destination directory (will be created).")
    s.set_defaults(func=cmd_paillier_export)

    s = sp.add_parser("he-paillier-selftest", help="Quick integer add/decrypt self-test.")
    s.set_defaults(func=cmd_paillier_selftest)

    #' CKKS
    s = sp.add_parser("he-ckks-init", help="Generate CKKS TenSEAL contexts.")
    s.add_argument("--poly-mod-degree", type=int, default=8192, help="Polynomial modulus degree (e.g., 8192).")
    s.add_argument("--coeff-mod-bits", default="60,40,40,60",
                   help="Comma/space-separated coefficient modulus bit sizes.")
    s.add_argument("--scale", default="2**40", help="Global scale, e.g., 2**40.")
    s.add_argument("--rotate", action="store_true", help="Rotate contexts (zeroize old files if present).")
    s.set_defaults(func=cmd_ckks_init)

    s = sp.add_parser("he-ckks-export", help="Export CKKS public context for clients.")
    s.add_argument("--dest", required=True, help="Destination directory (will be created).")
    s.set_defaults(func=cmd_ckks_export)

    s = sp.add_parser("he-ckks-selftest", help="Quick vector add/decrypt self-test.")
    s.set_defaults(func=cmd_ckks_selftest)

    return p

def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    #' Ensure he/ root exists + write manifest link if missing
    art = Path(args.artifact_dir)
    he_root = art / "he"
    he_root.mkdir(parents=True, exist_ok=True)
    if not (he_root / "manifest_link.json").exists():
        _link_manifest(he_root)
    #' Touch manifest to ensure Part_0 ran
    _manifest(art)
    args.func(args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
