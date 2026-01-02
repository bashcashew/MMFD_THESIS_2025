from __future__ import annotations

import argparse, hashlib, json, os, sys, tarfile, time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

__version__ = "0.7.0"

#' ====== IO HELPERS ======
def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")

def _read_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_json(p: Path, o: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(o, f, indent=2, sort_keys=True)

def _append(out_dir: Path, name: str, line: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / name, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")

def _sha256_file(p: Path, bufsize: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(bufsize)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _find_files(roots: List[Path], name: str) -> List[Path]:
    out: List[Path] = []
    for r in roots:
        for p in r.rglob(name):
            if p.is_file():
                out.append(p)
    out.sort()
    return out

def _discover_metrics(roots: List[Path]) -> List[Path]:
    return _find_files(roots, "metrics.json")

#' ====== SCOREBOARD ======
def _load_metric(file: Path) -> Optional[Dict[str, Any]]:
    try:
        j = _read_json(file)
        #' Try to establish normalized two shapes:
        #' - Part_3/7 centralized: flat metrics
        #' - Part_7 fl_clients: report with per_client + micro
        if "per_client" in j and "micro" in j:
            fam = j.get("family") or j.get("model") or "unknown"
            micro = j["micro"]
            return {
                "source": "fl_clients",
                "family": fam,
                "path": str(file),
                "AUPRC": micro.get("AUPRC"),
                "ROC_AUC": micro.get("ROC_AUC"),
                "F1": micro.get("F1"),
                "precision": micro.get("precision"),
                "recall": micro.get("recall"),
                "threshold": micro.get("threshold_used"),
                "notes": "micro aggregation of clients"
            }
        else:
            fam = j.get("family") or j.get("model") or "unknown"
            thr = j.get("threshold")
            if thr is None:
                thr = j.get("threshold_used")
            return {
                "source": "single",
                "family": fam,
                "path": str(file),
                "AUPRC": j.get("AUPRC"),
                "ROC_AUC": j.get("ROC_AUC"),
                "F1": j.get("F1"),
                "precision": j.get("precision"),
                "recall": j.get("recall"),
                "threshold": thr,
                "notes": j.get("threshold_source")
            }
    except Exception:
        return None

def _build_scoreboard(metric_files: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for m in metric_files:
        r = _load_metric(m)
        if r:
            rows.append(r)
    #' Sort by family then AUPRC desc
    rows.sort(key=lambda x: (str(x.get("family")), -(x.get("AUPRC") or 0.0)))
    return rows

#' ====== PACKAGE IT ======
def _collect_files_for_checksums(roots: List[Path]) -> List[Path]:
    files: List[Path] = []
    for r in roots:
        for p in r.rglob("*"):
            if p.is_file():
                files.append(p)
    files.sort()
    return files

def _make_tar_gz(out_path: Path, roots: List[Path], base_prefix: str = "") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w:gz") as tar:
        for r in roots:
            base = Path(base_prefix) / r.name if base_prefix else r.name
            tar.add(r, arcname=str(base))

#' ====== SANITIZATION ======
def _sanitize_manifest(in_path: Optional[Path], out_path: Path) -> Optional[Dict[str, Any]]:
    if not in_path or not in_path.exists():
        return None
    j = _read_json(in_path)
    #' Remove obvious secret-bearing sections (outside of zeroizing)
    def _drop(d: Dict[str, Any], key_path: List[str]) -> None:
        cur = d
        for k in key_path[:-1]:
            if k in cur and isinstance(cur[k], dict):
                cur = cur[k]
            else:
                return
        cur.pop(key_path[-1], None)

    secret_paths = [
        ["secrets"],
        ["psi"],
        ["he"],
        ["tls"],
        ["paths","raw_data"],
        ["debug","internal_env"]
    ]
    for pth in secret_paths:
        _drop(j, pth)
    _write_json(out_path, j)
    return j

#' ====== ZEROIZATION ======
def _best_effort_zeroize(path: Path) -> None:
    try:
        from crypto.zeroize import best_effort_wipe
        best_effort_wipe(path)
    except Exception:
        try:
            if path.exists() and path.is_file():
                sz = path.stat().st_size
                with open(path, "r+b") as f:
                    f.write(b"\x00" * sz)
                    f.flush(); os.fsync(f.fileno())
                path.unlink(missing_ok=True)
        except Exception:
            pass

def _glob_all(roots: List[Path], patterns: List[str]) -> List[Path]:
    out: List[Path] = []
    for r in roots:
        for pat in patterns:
            out.extend(p for p in r.rglob(pat) if p.is_file())
    #' de-dup & sort
    norm = sorted(set(out))
    return norm

def _zeroize_targets(out_dir: Path, roots: List[Path],
                     he: bool, psi: bool, sag: bool, certs: bool) -> List[str]:
    removed: List[str] = []

    if he:
        #' Paillier private & CKKS secret context
        he_targets = _glob_all(roots, [
            "he/paillier/private.json",
            "he/ckks/context_server.bin"
        ])
        for p in he_targets:
            _best_effort_zeroize(p); removed.append(str(p))
            _append(out_dir, "zeroize_log.txt", f"[HE] wiped {p}")

    if psi:
        psi_targets = _glob_all(roots, [
            "psi/hmac_key.json",
            "**/hmac_key.bin",
            "**/ids_hmac_map.csv",
            "psi/**/*.key", "psi/**/*.salt",
            "psi/**/*.tokens", "psi/**/*.txt",
            "oprf/**/*.json", "oprf/**/*.bin"
        ])
        for p in psi_targets:
            _best_effort_zeroize(p); removed.append(str(p))
            _append(out_dir, "zeroize_log.txt", f"[PSI] wiped {p}")

    if sag:
        sag_targets = _glob_all(roots, [
            "secureagg/**/mask.npy",
            "secureagg/**/pairwise_secrets.json",
            "secureagg/**/commit.json"
        ])
        for p in sag_targets:
            _best_effort_zeroize(p); removed.append(str(p))
            _append(out_dir, "zeroize_log.txt", f"[SECUREAGG] wiped {p}")

    if certs:
        cert_targets = _glob_all(roots, [
            "tools/**/server.key", "tools/**/client.key", "tools/**/ca.key",
            "**/*.pem.key", "**/*.p12", "**/*.pfx"
        ])
        for p in cert_targets:
            _best_effort_zeroize(p); removed.append(str(p))
            _append(out_dir, "zeroize_log.txt", f"[CERTS] wiped {p}")

    return removed


#' ====== CLI ======
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Part_8: Finalize run (report, checksums, optional package & zeroize)."
    )
    p.add_argument("--include", type=Path, nargs="+", required=True,
                   help="Root directories to include for report/packaging (explicit, no guessing).")
    p.add_argument("--manifest", type=Path, default=None,
                   help="Optional Part_0 run_manifest.json to sanitize & re-emit.")
    p.add_argument("--out", type=Path, required=True, help="Output directory for final artifacts.")

    #' Packaging & checksums
    p.add_argument("--package", choices=["none","tar.gz"], default="none",
                   help="Optional archive format (default: none).")
    p.add_argument("--emit-checksums", action="store_true",
                   help="Write checksums.txt for all files under --include roots.")

    #' Security options
    p.add_argument("--zeroize-he", action="store_true", help="Wipe HE server-side secrets (Paillier private, CKKS secret).")
    p.add_argument("--zeroize-psi", action="store_true", help="Wipe PSI keys/tokens (HMAC/OPRF).")
    p.add_argument("--zeroize-secureagg", action="store_true", help="Wipe secure-aggregation masks/dev secrets.")
    p.add_argument("--zeroize-certs", action="store_true", help="Wipe private keys for dev TLS/mTLS certs.")



#'====== MAIN ======
def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    includes: List[Path] = [p.resolve() for p in args.include]

    metric_files = _discover_metrics(includes)
    scoreboard = _build_scoreboard(metric_files)
    report = {
        "version": __version__,
        "generated_at": _now(),
        "includes": [str(p) for p in includes],
        "metrics_files": [str(p) for p in metric_files],
        "scoreboard": scoreboard
    }
    _write_json(out_dir / "run_report.json", report)


    if args.manifest:
        sanitized = _sanitize_manifest(args.manifest, out_dir / "run_manifest_public.json")
        if sanitized is not None:
            _append(out_dir, "run_report.log", "[manifest] sanitized run_manifest.json → run_manifest_public.json")



    if args.emit_checksums or args.package != "none":
        files = _collect_files_for_checksums(includes)
        with open(out_dir / "checksums.txt", "w", encoding="utf-8") as f:
            for p in files:
                try:
                    h = _sha256_file(p)
                    f.write(f"{h}  {p}\n")
                except Exception:
                    continue


    if args.package == "tar.gz":
        pkg = out_dir / "package.tar.gz"
        _make_tar_gz(pkg, includes)
        _append(out_dir, "run_report.log", f"[package] wrote {pkg}")


    if any([args.zeroize_he, args.zeroize_psi, args.zeroize_secureagg, args.zeroize_certs]):
        removed = _zeroize_targets(out_dir, includes,
                                   he=args.zeroize_he,
                                   psi=args.zeroize_psi,
                                   sag=args.zeroize_secureagg,
                                   certs=args.zeroize_certs)
        _append(out_dir, "run_report.log", f"[zeroize] removed {len(removed)} files; see zeroize_log.txt")

    _append(out_dir, "run_report.log", "[finalize] done.")
    print(f"[Part_8:{__version__}] Finalized. Report → {out_dir/'run_report.json'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
