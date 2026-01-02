from __future__ import annotations

import argparse, base64, io, json, os, ssl, sys, threading, time
import contextlib
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple, Set

import numpy as np


try:
    from joblib import dump, load
except Exception:
    dump = load = None

try:
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest
except Exception:
    RandomForestClassifier = HistGradientBoostingClassifier = IsolationForest = None

#' aggregators (amassed after PARTS in models/fl/aggregators.py)
try:
    from models.fl.aggregators import (
        FedAvgLR, FedForestUnion, FedHGBEnsemble, FedIForestEnsemble, FedGBMSSRFEnsemble
    )
except Exception as e:
    FedAvgLR = FedForestUnion = FedHGBEnsemble = FedIForestEnsemble = FedGBMSSRFEnsemble = None


__version__ = "0.7.1"

MAX_BODY_BYTES = int(os.environ.get("FFD_FL_MAX_BODY", str(50 * 1024 * 1024)))  # 50MB


#' ====== UTILITIES ======
def _write_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def _read_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _b64encode_bytes(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")

def _b64decode_to_bytes(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))

def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")

def _now_ms() -> int:
    return int(time.time() * 1000)

def _log(out_dir: Path, *msgs: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "logs.txt", "a", encoding="utf-8") as f:
        for m in msgs:
            f.write(m.rstrip() + "\n")
    print(*msgs, flush=True)


#' ======= SERVER STATE STATATUS ======
class FLServerState:
    def __init__(self, *, model: str, clients_expected: int, rounds: int,
                 he: str, he_dir: Optional[Path], lr_dim: Optional[int],
                 out_dir: Path, debug:bool):
        self.model = model
        self.clients_expected = int(clients_expected)
        self.rounds_total = int(rounds)
        self.he = he
        self.he_dir = he_dir
        self.lr_dim = lr_dim  #' if None/auto, we expect first LR client to supply
        self.out_dir = out_dir

        self.round = 1
        self.registered: Dict[str, Dict[str, Any]] = {}
        self.updates: Dict[int, List[Tuple[str, Dict[str, Any]]]] = {}
        self.seen_clients: Dict[int, Set[str]] = {}
        self.global_model: Optional[Any] = None  #' theta (LR) or sklearn model/meta
        self.aggregator: Optional[Any] = None
        self.lock = threading.Lock()
        self.finished: bool = False
        self.debug: bool = bool(debug)

        self._save_state()

    #' Persistence
    def _round_dir(self, k: int) -> Path:
        d = self.out_dir / "rounds" / f"{k:04d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save_state(self) -> None:
        state = {
            "version": __version__,
            "model": self.model,
            "round": self.round,
            "clients_expected": self.clients_expected,
            "registered": list(self.registered.keys()),
            "rounds_total": self.rounds_total,
            "he": self.he,
            "lr_dim": self.lr_dim,
            "have_global": self.global_model is not None,
            "finished": self.finished
        }
        _write_json(self.out_dir / "server_state.json", state)

    #' ====== REGISTRATION ======
    def register(self, client_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            self.registered.setdefault(client_id, {
                "since": _ts(), "model": payload.get("model")
            })
            #' model sanity
            if payload.get("model") and str(payload.get("model")) != self.model:
                raise ValueError(f"Server model is '{self.model}', client provided '{payload.get('model')}'.")
            #' If LR and dim=auto: accept dim from client
            if self.model == "lr" and (self.lr_dim is None or self.lr_dim == 0):
                dim = int(payload.get("dim", 0))
                if dim <= 0:
                    raise ValueError("LR requires 'dim' on /register when --lr-dim=auto.")
                self.lr_dim = dim
            self._save_state()
            return {"ok": True, "round": self.round, "expected_clients": self.clients_expected}

    #' ====== mark a client as "received" (even with no contribution) ======
    def mark_received(self, client_id: str) -> None:
        with self.lock:
            self.seen_clients.setdefault(self.round, set()).add(client_id)
            self._save_state()

    #' ====== aggregator go ======
    def _ensure_aggregator_for_round(self) -> None:
        if self.aggregator is not None:
            return
        if any(a is None for a in [FedAvgLR, FedForestUnion, FedHGBEnsemble, FedIForestEnsemble, FedGBMSSRFEnsemble]):
            raise RuntimeError("models.fl.aggregators not available; ensure models/fl/aggregators.py is importable.")
        if self.model in ("rf","hgb","iforest","gbm_ssrf") and load is None:
            raise RuntimeError("joblib / sklearn not available to (de)serialize models for tree/GBM paths.")
        if self.model == "lr":
            if self.lr_dim is None or self.lr_dim <= 0:
                raise RuntimeError("LR aggregator needs --lr-dim or client-provided dim on /register.")
            #' Create LR FedAvg with chosen HE scheme
            agg = FedAvgLR(dim=self.lr_dim, lr=0.1, he_scheme=self.he, he_dir=str(self.he_dir) if self.he_dir else None)
            agg.theta = np.zeros(self.lr_dim, dtype=np.float64)
            self.aggregator = agg
        elif self.model == "rf":
            self.aggregator = FedForestUnion(max_trees=800)
        elif self.model == "hgb":
            self.aggregator = FedHGBEnsemble()
        elif self.model == "iforest":
            self.aggregator = FedIForestEnsemble()
        elif self.model == "gbm_ssrf":
            self.aggregator = FedGBMSSRFEnsemble()
        else:
            raise RuntimeError(f"Unknown model: {self.model}")

    #' ====== UPDATES ======
    def add_update(self, client_id: str, round_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        if client_id not in self.registered:
            raise ValueError("Client not registered; call /register first.")
        if round_id != self.round:
            raise ValueError(f"Stale or future round: got {round_id}, server at {self.round}")
        with self.lock:
            self.updates.setdefault(self.round, [])
            self.seen_clients.setdefault(self.round, set())
            if client_id in self.seen_clients[self.round]:
                raise ValueError(f"Duplicate update from client '{client_id}' for round {self.round}.")
            self.updates[self.round].append((client_id, payload))
            self.seen_clients[self.round].add(client_id)
            got = len(self.updates[self.round])

            if self.debug:
                rdir = self._round_dir(self.round)
                inbox = self.out_dir / "inbox" / f"{self.round:04d}"
                inbox.mkdir(parents=True, exist_ok=True)
                _write_json(inbox / f"{client_id}.json", {"client_id": client_id, "payload_keys": list(payload.keys())})
            self._save_state()
            return {"ok": True, "received": got, "need": self.clients_expected}

    #' ====== FINALIZE ROUNDS ======
    def maybe_finalize_round(self) -> Optional[Dict[str, Any]]:
        with self.lock:
            if len(self.seen_clients.get(self.round, set())) < self.clients_expected:
                return None  #' waits for more clients
            #' 1.Aggregation
            self._ensure_aggregator_for_round()
            agg = self.aggregator
            agg.begin_round()

            #' 2.Feed client updates
            for client_id, payload in self.updates.get(self.round, []):
                mdl = self.model

                #' 3. Skip clients that had no local TRAIN rows (n==0)
                if int(payload.get("n", 0)) == 0:
                    continue

                if mdl == "lr":
                    #' payload: {"n": int, "grad_sum": ...}
                    #' grad_sum may be list[float] (he=none) or base64 bytes (he=paillier/ckks)
                    g = payload["grad_sum"]
                    if self.he == "none":
                        grad = np.asarray(g, dtype=np.float64)
                    else:
                        #' Pass-through cipher object: expects crypto loaders to be used by HEAdapter.add_cipher_sums
                        grad = _b64decode_to_bytes(g) if isinstance(g, str) else g
                    agg.accept_update(client_id, {"n": int(payload["n"]), "grad_sum": grad})
                elif mdl in ("rf", "hgb", "iforest", "gbm_ssrf"):
                    model_b64 = payload["model_b64"]
                    model_bytes = _b64decode_to_bytes(model_b64)
                    #' Deserialize sklearn model/dict
                    buf = io.BytesIO(model_bytes)
                    model_obj = load(buf)
                    agg.accept_update(client_id, {"model": model_obj, "n": int(payload.get("n", 0))})
                else:
                    raise RuntimeError(f"Unknown model: {mdl}")

            #' Finalize
            t0 = _now_ms()
            global_obj = agg.finalize_round()
            dt = _now_ms() - t0

            #' Persist artifacts
            rdir = self._round_dir(self.round)
            summary = {
                "round": self.round,
                "model": self.model,
                "clients": [c for c, _ in self.updates.get(self.round, [])],
                "received": len(self.updates.get(self.round, [])),
                "finalize_ms": dt,
                "he": self.he
            }

            if self.model == "lr":
                theta = np.asarray(global_obj, dtype=np.float64).tolist()
                _write_json(rdir / "theta.json", {"theta": theta, "lr": 0.1})
                self.global_model = {"theta": theta}
            else:
                buf = io.BytesIO()
                dump(global_obj, buf)
                (rdir / "global_model.joblib").write_bytes(buf.getvalue())
                self.global_model = {"path": str(rdir / "global_model.joblib")}

            _write_json(rdir / "agg_summary.json", summary)
            self._save_state()

            #' Advance rounds
            if self.round >= self.rounds_total:
                self.finished = True
                self._save_state()
                return {"ok": True, "finalized": summary, "next_round": None, "finished": True}
            else:
                self.round += 1
                self.aggregator = None
                return {"ok": True, "finalized": summary, "next_round": self.round, "finished": False}


#' ====== HTTP LAYER ======
class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class FLRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _reject_if_finished(self):
        if self.state.finished:
            self._send(409, {"error": "training finished"}); return True
        return False

    def _json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_BODY_BYTES:
            _ = self.rfile.read(min(length, 1024))
            self.send_response(413); self.end_headers()
            return {}
        data = self.rfile.read(length) if length > 0 else b"{}"
        try:
            return json.loads(data.decode("utf-8"))
        except Exception:
            return {}

    def _send(self, code: int, obj: Dict[str, Any]):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @property
    def state(self) -> FLServerState:
        return self.server.state

    def do_GET(self):
        try:
            if self.path.startswith("/status"):
                self._send(200, _read_json(self.state.out_dir / "server_state.json"))
                return
            if self.path.startswith("/global"):
                #' Return global for #round=<int|latest>
                qs = parse_qs(urlparse(self.path).query or "")
                #' robust k resolution with safe default
                k = max(1, self.state.round - 1 if not self.state.finished else self.state.rounds_total)
                if "round" in qs and qs["round"]:
                    req = (qs["round"][0] or "").strip().lower()
                    if req == "latest":
                        pass
                    else:
                        with contextlib.suppress(Exception):
                            k = max(1, int(req))
                rdir = self.state._round_dir(k)
                res = {"round": k, "model": self.state.model, "have_global": False}
                if self.state.model == "lr" and (rdir / "theta.json").exists():
                    res["have_global"] = True
                    res["theta"] = _read_json(rdir / "theta.json")["theta"]
                elif (rdir / "global_model.joblib").exists():
                    res["have_global"] = True
                    res["model_path"] = str(rdir / "global_model.joblib")
                self._send(200, res); return
            if self.path.startswith("/health"):
                self._send(200, {"ok": True, "time": _ts()}); return
            self._send(404, {"error": "not found"})
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        try:
            if self.path.startswith("/register"):
                if self._reject_if_finished(): return
                body = self._json()
                ack = self.state.register(body["client_id"], body)
                self._send(200, ack); return

            if self.path.startswith("/update"):
                if self._reject_if_finished(): return
                body = self._json()

                n_local = int(body.get("n", 0))
                client_id = body.get("client_id","?")
                model = body.get("model","?")
                if n_local == 0:
                    if hasattr(self, "state") and hasattr(self.state, "mark_received"):
                        self.state.mark_received(client_id)
                    elif hasattr(self, "round_state"):
                        self.round_state.setdefault("received", set()).add(client_id)

                    fin = self.state.maybe_finalize_round()
                    if fin:
                        self._send(200, fin)
                        if fin.get("finished"):
                            threading.Thread(target=self.server.shutdown, daemon=True).start()
                    else:
                        self._send(200, {"ok": True, "skipped": True,
                                         "reason": "n==0 (empty local TRAIN)"})
                    return

                ack = self.state.add_update(body["client_id"], int(body["round"]), body)
                fin = self.state.maybe_finalize_round()
                if fin:
                    self._send(200, fin)
                    if fin.get("finished"):
                        threading.Thread(target=self.server.shutdown, daemon=True).start()
                else:
                    self._send(200, ack)
                return

            self._send(404, {"error": "not found"})
        except Exception as e:
            self._send(400, {"error": f"{type(e).__name__}: {e}"})


#' ====== TLS HELPERS ======
def build_ssl_context(tls_mode: str, cert_dir: str):
    """
    Create an SSLContext for TLS or mTLS.

    Fixes TLS 1.3 cipher selection on macOS/LibreSSL/OpenSSL 3 by:
    - Using set_ciphersuites() for TLS 1.3 (if present),
    - Using set_ciphers() for TLS 1.2 fallback only,
    - Enforcing TLSv1.2+ and disabling compression.
    """
    import os, ssl

    if tls_mode not in ("tls", "mtls"):
        return None

    ca = os.path.join(cert_dir, "ca.crt")
    crt = os.path.join(cert_dir, "server.crt")
    key = os.path.join(cert_dir, "server.key")

    if not (os.path.isfile(ca) and os.path.isfile(crt) and os.path.isfile(key)):
        raise RuntimeError(f"[tls] Missing CA/server cert or key in {cert_dir}")

    #' Modern protocol selector
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

    #' Enforce TLSv1.2+ and disable compression
    try:
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    except Exception:
        pass
    try:
        ctx.options |= getattr(ssl, "OP_NO_COMPRESSION", 0)
    except Exception:
        pass

    #' Load certs
    ctx.load_cert_chain(certfile=crt, keyfile=key)
    ctx.load_verify_locations(cafile=ca)

    #' Require client certs for mTLS
    if tls_mode == "mtls":
        ctx.verify_mode = ssl.CERT_REQUIRED
        try:
            ctx.check_hostname = False
        except Exception:
            pass

    #' ====== Cipher configuration ======
    try:
        if hasattr(ctx, "set_ciphersuites"):
            ctx.set_ciphersuites(
                "TLS_AES_256_GCM_SHA384:"
                "TLS_CHACHA20_POLY1305_SHA256:"
                "TLS_AES_128_GCM_SHA256"
            )
    except Exception:
        pass

    try:
        ctx.set_ciphers(
            "ECDHE+AESGCM:"
            "ECDHE+CHACHA20:"
            "ECDHE+AES:"
            "!aNULL:!eNULL:!MD5:!RC4"
        )
    except Exception:
        pass

    return ctx


#' ====== CLI ======
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Part_4: HFL Server with TLS/mTLS and HE-aware aggregation."
    )
    p.add_argument("--model", choices=["lr","rf","hgb","gbm_ssrf","iforest"], required=True)
    p.add_argument("--clients", type=int, required=True, help="Expected number of clients per round.")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--debug", action="store_true", help="Cache minimal payload metadata to inbox/ for debugging.")

    #' LR specifics
    p.add_argument("--lr-dim", type=str, default="auto", help="'auto' (first client provides) or integer.")
    p.add_argument("--he", choices=["none","paillier","ckks"], default="none")
    p.add_argument("--he-dir", type=Path, default=None, help="HE artifacts dir (Part_1B outputs).")

    #' Network
    p.add_argument("--bind", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8842)
    p.add_argument("--tls", choices=["off","tls","mtls"], default="tls")
    p.add_argument("--cert-dir", type=Path, default=Path("tools/certs"))

    #' Artifacts
    p.add_argument("--out", type=Path, required=True)
    return p

def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = args.out; out_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "lr":
        if str(args.lr_dim).lower() == "auto":
            lr_dim: Optional[int] = None
        else:
            try:
                lr_dim = int(args.lr_dim)
            except Exception:
                raise SystemExit("--lr-dim must be 'auto' or an integer.")
        if args.he != "none" and not args.he_dir:
            raise SystemExit("LR with HE requires --he-dir pointing to Part_1B outputs.")
    else:
        lr_dim = 0

    state = FLServerState(
        model=args.model, clients_expected=args.clients, rounds=args.rounds,
        he=args.he, he_dir=args.he_dir, lr_dim=lr_dim, out_dir=out_dir, debug=args.debug
    )

    # HTTP server
    httpd = _ThreadingHTTPServer((args.bind, int(args.port)), FLRequestHandler)
    httpd.state = state
    ssl_ctx = build_ssl_context(args.tls, args.cert_dir)
    if ssl_ctx:
        httpd.socket = ssl_ctx.wrap_socket(httpd.socket, server_side=True)
        _log(out_dir, f"[Part_4:{__version__}] TLS={args.tls} cert_dir={args.cert_dir}")
    else:
        _log(out_dir, f"[Part_4:{__version__}] TLS=off (dev only)")

    _log(out_dir, f"[Part_4:{__version__}] model={args.model} clients={args.clients} rounds={args.rounds} "
                  f"bind={args.bind}:{args.port} HE={args.he} lr_dim={state.lr_dim or 'auto'}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        _log(out_dir, "[Part_4] Shutting down…")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
