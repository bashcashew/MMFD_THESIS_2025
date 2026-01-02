from __future__ import annotations

import argparse, json, os, sys, platform, importlib.util, time, random, string
from pathlib import Path
from typing import Dict, Any, Optional, List


__version__ = "0.6.2"
#' ====== HELPERS ======
def _has_lib(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None

def detect_capabilities() -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libs": {
            "polars": _has_lib("polars"),
            "torch": _has_lib("torch"),
            "phe": _has_lib("phe"),
            "tenseal": _has_lib("tenseal"),
            "pyfhel": _has_lib("Pyfhel"),
            "cryptography": _has_lib("cryptography"),
            "oprf": _has_lib("oprf"),
            "sklearn": _has_lib("sklearn"),
            "joblib": _has_lib("joblib"),
        },
    }

def recommended_thread_count(user_req: str | int = "auto") -> int:
    total = max(1, (os.cpu_count() or 1))
    if isinstance(user_req, str) and user_req == "auto":
        return max(1, total - 1)
    try:
        n = int(user_req); return max(1, min(total, n))
    except Exception:
        return max(1, total - 1)

def set_thread_env(n_threads: int) -> Dict[str, str]:
    env = {
        "OMP_NUM_THREADS": str(n_threads),
        "OPENBLAS_NUM_THREADS": str(n_threads),
        "MKL_NUM_THREADS": str(n_threads),
        "VECLIB_MAXIMUM_THREADS": str(n_threads),
        "NUMEXPR_NUM_THREADS": str(n_threads),
        "NUMEXPR_MAX_THREADS": str(n_threads),
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    }
    for k, v in env.items():
        os.environ[k] = v
    return env

def _run_id() -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    rnd = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{ts}-{rnd}"

#' ====== PRESETS ======
def load_preset(name: str) -> Dict[str, Any]:
    n = (name or "none").lower()
    if n == "paysim":
        return {
            "dataset": {
                "preset": "paysim",
                "target": "isFraud",
                "id_columns": ["nameOrig", "nameDest"],
                "feature_policy": {"include": [], "exclude": ["isFlaggedFraud"]},
            },
            "preprocessing": {"class_weighting": "class_balanced", "engine": "pandas"},
        }
    if n == "research_grade":
        #' Same target/IDs, but advanced downstream recipe enabled by default.
        return {
            "dataset": {
                "preset": "paysim",
                "target": "isFraud",
                "id_columns": ["nameOrig", "nameDest"],
                "feature_policy": {"include": [], "exclude": ["isFlaggedFraud"]},
            },
            "preprocessing": {"class_weighting": "class_balanced", "engine": "pandas",
                              "recipe": {"name": "research-grade", "config": None}},
        }
    return {"dataset": {"preset": "none"}, "preprocessing": {"engine": "pandas"}}

#' ====== CLI ARGS ======
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="Part_0",
        description="FFD bootstrap: threads, capability checks, mode selection, and manifest."
    )
    p.add_argument("--mode", choices=["centralized","hfl","vfl","hybrid"], default="centralized")
    #' Data: official flag + backward-comp. aliases
    p.add_argument("--data", nargs="+", help="Path(s) to dataset CSV/Parquet.")
    p.add_argument("--input", help="(alias) single dataset path; will be included in dataset.paths.")
    #' Preset / target / engine / weighting
    p.add_argument("--preset", choices=["none","paysim","research_grade"], default="none", help="Dataset preset.")
    p.add_argument("--target", default=None, help="Override target column (else from preset).")
    p.add_argument("--engine", choices=["pandas","polars"], default=None, help="Preprocessing engine hint.")
    p.add_argument("--class-weighting",
        choices=["none","balanced","class_balanced","inverse_global","both"],
        default="none",
        help="Sample weighting strategy: "
        "'class_balanced' (alias: 'balanced'), 'inverse_global', 'both', or 'none'.")

    #' Recipe selection & optional config path
    p.add_argument("--recipe", choices=["none","research-grade"], default="none",
                   help="Optional advanced preprocessing recipe.")
    p.add_argument("--recipe-config", default=None,
                   help="Path to YAML/JSON config for the selected recipe (optional).")

    #' Models (explicit parity across centralized and FL)
    p.add_argument("--models", default="lr,rf,hgb,gbm_ssrf,iforest",
                   help="Comma-separated list: lr,rf,hgb,gbm_ssrf,iforest")

    #' Crypto / security
    p.add_argument("--psi", choices=["none","hmac","oprf"], default="none")
    p.add_argument("--he", choices=["none","paillier","ckks"], default="none")
    p.add_argument("--zeroize", action="store_true")
    p.add_argument("--tls", choices=["off","tls","mtls"], default="off")
    #' TLS file args (for FL runners); centralized will ignore
    p.add_argument("--cafile"); p.add_argument("--certfile"); p.add_argument("--keyfile"); p.add_argument("--keypass")
    #' Legacy helper
    p.add_argument("--cert-dir", default="tools/certs")

    #' FL knobs
    p.add_argument("--clients", type=int, default=2)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--aggregator", choices=["fedavg","fedprox"], default="fedavg")

    #' System / bookkeeping
    p.add_argument("--threads", default="auto")
    p.add_argument("--artifact-dir", default="artifacts/run")
    p.add_argument("--out", help="(alias) artifact dir; same as --artifact-dir")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug", action="store_true")
    return p

#' ====== ASSEMBLY ======
def _parse_models(s: str) -> List[str]:
    allowed = {"lr","rf","hgb","gbm_ssrf","iforest"}
    req = [m.strip().lower() for m in s.split(",") if m.strip()]
    bad = [m for m in req if m not in allowed]
    if bad:
        raise SystemExit(f"Unknown model(s): {bad}. Allowed: {sorted(allowed)}")
    #' maintain given order without dups
    seen, out = set(), []
    for m in req:
        if m not in seen:
            out.append(m); seen.add(m)
    return out

def _ckks_backend() -> Optional[str]:
    if _has_lib("tenseal"): return "tenseal"
    if _has_lib("Pyfhel"): return "pyfhel"
    return None

def _normalize_weighting(w: Optional[str]) -> str:
    """
    Map legacy/alias values to canonical names used by preprocessing:
      - 'balanced' -> 'class_balanced'
    """
    m = (w or "none").lower()
    if m == "balanced":
        return "class_balanced"
    return m

def _coerce_mode_posture(ns) -> None:
    """Mode-aware guardrails: centralized ignores PSI/HE/TLS; FL validates TLS args lightly."""
    if ns.mode == "centralized":
        #' Hard-coerce to no-crypto for local baseline clarity
        ns.psi = "none"; ns.he = "none"; ns.tls = "off"
        return
    #' FL: if TLS requested, ensure server-side materials exist (clients checked later)
    if ns.tls in {"tls","mtls"}:
        missing = [k for k in ("cafile","certfile","keyfile") if not getattr(ns, k, None)]
        if missing:
            raise SystemExit(f"--tls {ns.tls} requires: " + ", ".join(f"--{m}" for m in missing))
    #' CKKS presence note
    if ns.he == "ckks" and _ckks_backend() is None:
        print("[Part_0] WARNING: --he ckks selected but no TenSEAL/Pyfhel available; "
              "downstream may switch posture or skip HE.", file=sys.stderr)

def assemble_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    caps = detect_capabilities()
    preset_cfg = load_preset(args.preset)
    enabled_models = _parse_models(args.models)

    #' dataset paths: prefer --data, else --input, else []
    paths: List[str] = []
    if args.data:
        paths.extend(args.data)
    if args.input:
        paths.append(args.input)

    #' derive target/engine/weighting/recipe from preset unless explicitly overridden
    target = args.target or preset_cfg.get("dataset", {}).get("target")
    engine = args.engine or preset_cfg.get("preprocessing", {}).get("engine", "pandas")
    class_weighting = args.class_weighting if args.class_weighting != "none" \
        else preset_cfg.get("preprocessing", {}).get("class_weighting", "none")
    #' normalize aliases to match preprocessing module API ('class_balanced'/'inverse_global'/'both'/'none')
    class_weighting = _normalize_weighting(class_weighting)
    recipe_name = args.recipe
    recipe_cfg = args.recipe_config
    #' if preset preconfigures research-grade and user didn't override, inherit it
    preset_recipe = preset_cfg.get("preprocessing", {}).get("recipe")
    if preset_recipe and args.recipe == "none":
        recipe_name = preset_recipe.get("name", "research-grade")
        recipe_cfg = preset_recipe.get("config")

    dataset = {
        "paths": paths,
        "preset": preset_cfg.get("dataset", {}).get("preset", "none"),
        "target": target,
        "id_columns": preset_cfg.get("dataset", {}).get("id_columns", []),
        "feature_policy": preset_cfg.get("dataset", {}).get("feature_policy", {"include": [], "exclude": []}),
    }

    preprocessing = {
        "engine": engine,
        "class_weighting": class_weighting,
        "recipe": { "name": recipe_name, "config": recipe_cfg },
    }

    #' threads
    eff_threads = recommended_thread_count(args.threads)
    env_caps = set_thread_env(eff_threads)

    manifest = {
        "version": __version__,
        "run_id": _run_id(),
        "mode": args.mode,
        "models": {
            "enabled": enabled_models,
            "fl": enabled_models
        },
        "dataset": dataset,
        "preprocessing": preprocessing,
        "crypto": {"psi": args.psi, "he": args.he, "zeroize": bool(args.zeroize)},
        "tls": {
            "mode": args.tls,
            "cafile": args.cafile, "certfile": args.certfile, "keyfile": args.keyfile, "keypass": args.keypass
        },
        "fl": {"clients": int(args.clients), "rounds": int(args.rounds), "aggregator": args.aggregator},
        "threads": {"requested": args.threads, "effective": eff_threads},
        "seed": int(args.seed),
        "capabilities": caps,
        "threads_env": env_caps,
    }
    return manifest

def write_artifacts(manifest: Dict[str, Any], artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with open(artifact_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    with open(artifact_dir / "env_threads.json", "w", encoding="utf-8") as f:
        json.dump(manifest["threads"], f, indent=2, sort_keys=True)
    with open(artifact_dir / "capabilities.json", "w", encoding="utf-8") as f:
        json.dump(manifest["capabilities"], f, indent=2, sort_keys=True)

    print(f"[Part_0:{__version__}] Wrote manifest: {artifact_dir/'run_manifest.json'}")
    print(f"[Part_0:{__version__}] Mode={manifest['mode']}  Preset={manifest['dataset']['preset']}  "
          f"Recipe={manifest['preprocessing']['recipe']['name']}  Target={manifest['dataset']['target']}")
    print(f"[Part_0:{__version__}] Models={','.join(manifest['models']['enabled'])}  "
          f"PSI={manifest['crypto']['psi']}  HE={manifest['crypto']['he']}  TLS={manifest['tls']['mode']}  "
          f"Threads={manifest['threads']['effective']}  Seed={manifest['seed']}")

#' ====== MAIN ======
def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    #' Back-compat for runners that pass --out instead of --artifact-dir
    if args.out:
        args.artifact_dir = args.out

    #' Guardrails per mode / posture
    _coerce_mode_posture(args)

    if args.mode in ("hfl", "hybrid") and args.clients < 2:
        print("[WARN] FL mode requested but --clients < 2; bumping to 2.")
        args.clients = 2
    if args.tls == "mtls" and not (args.cafile and args.certfile and args.keyfile):
        #' Soft warning; server/client parts will fail hard if truly required.
        print(f"[WARN] mTLS enabled but some TLS files are missing (cafile/certfile/keyfile).", file=sys.stderr)

    manifest = assemble_manifest(args)
    write_artifacts(manifest, Path(args.artifact_dir))

    print("\nNext steps:")
    if manifest["crypto"]["psi"] != "none":
        print("  1) Part_1A_psi.py  (PSI tokenization & intersection).")
    if manifest["crypto"]["he"] != "none":
        print("  2) Part_1B_he_keys.py  (HE keys/contexts).")
    print("  3) Part_2_preprocess.py  (preset/recipe; class weighting; PSI-aware).")
    if manifest["mode"] == "centralized":
        print("  4) Part_3_train_centralized.py  (centralized LR/RF/HGB/GBM-SSRF/IF).")
    else:
        print("  4) Part_4_fl_server.py & Part_5_fl_client.py  (HFL/VFL/Hybrid).")
        print("  5) Part_6_aggregate.py  (if separate secure aggregation is enabled).")
    print("  6) Part_7_evaluate.py  (end-of-pipeline metrics).")
    print("  7) Part_8_finalize.py  (packaging, zeroization).")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
