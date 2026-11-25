#!/usr/bin/env python3
"""Minimal standard-PD proxy: client -> (prefill, decode) with a KV hand-off.

Why this file exists: the in-tree example (`examples/disaggregated/
disaggregated_serving/disagg_proxy_demo.py`) never sets `kv_transfer_params` on
the prefill hop and never forwards the prefill response's `kv_transfer_params`
to decode, so with `MooncakeConnectorV1` no KV is ever transferred (the
connector only produces/consumes KV when those params are present:
`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py:831`).

The structure mirrors `vllm/v1/lwd_control/pd/proxy.py` (the LWD proxy that is
already curl-verified on this cluster) minus the LWD envelope:
  1. POST the client body to PREFILL with max_tokens=1 and
     kv_transfer_params={"do_remote_decode": true, "do_remote_prefill": false};
  2. read `kv_transfer_params` from the prefill response;
  3. POST the original body to DECODE with those params and relay the answer.

stdlib only, so it runs on the servers without extra packages.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error as urlerror
from urllib import request as urlrequest

# No proxy: the hop is host-local; urllib would otherwise honour http_proxy.
OPENER = urlrequest.build_opener(urlrequest.ProxyHandler({}))

LOCK = threading.Lock()
STATS = {"ok": 0, "failed": 0}


def log(message: str) -> None:
    sys.stderr.write(f"[pd-proxy] {message}\n")
    sys.stderr.flush()


def post(url: str, body: dict, timeout: float) -> tuple[int, bytes]:
    data = json.dumps(body).encode("utf-8")
    request = urlrequest.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with OPENER.open(request, timeout=timeout) as response:
            return response.status, response.read()
    except urlerror.HTTPError as exc:
        return exc.code, exc.read()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "pd-proxy/1.0"

    # -- plumbing ---------------------------------------------------------
    def log_message(self, fmt: str, *args) -> None:  # keep the log readable
        log(fmt % args)

    def _send(self, status: int, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, status: int, value: dict) -> None:
        self._send(status, json.dumps(value).encode("utf-8"), "application/json")

    def _routes(self) -> dict:
        return {
            "/v1/chat/completions": "/v1/chat/completions",
            "/v1/completions": "/v1/completions",
        }

    # -- GET --------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        if path in ("/health", "/healthz"):
            self._send_json(200, {"status": "ok", "stats": dict(STATS)})
            return
        if path == "/v1/models":
            # models are identical on both engines; ask prefill.
            try:
                request = urlrequest.Request(f"{self.server.prefill}/v1/models")
                with OPENER.open(request, timeout=self.server.timeout) as response:
                    status, payload = response.status, response.read()
            except urlerror.HTTPError as exc:
                status, payload = exc.code, exc.read()
            except Exception as exc:  # noqa: BLE001
                self._send_json(502, {"error": f"prefill /v1/models failed: {exc!r}"})
                return
            self._send(status, payload, "application/json")
            return
        self._send_json(404, {"error": f"unknown path {self.path}"})

    # -- POST -------------------------------------------------------------
    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        suffix = self._routes().get(path)
        if suffix is None:
            self._send_json(404, {"error": f"unknown path {self.path}"})
            return

        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw or b"{}")
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
        except Exception as exc:  # noqa: BLE001
            self._send_json(400, {"error": f"invalid JSON body: {exc}"})
            return

        started = time.monotonic()
        try:
            self._handoff(suffix, body)
        except Exception as exc:  # noqa: BLE001 - never leak a traceback to the client
            with LOCK:
                STATS["failed"] += 1
            log(f"request failed after {time.monotonic() - started:.1f}s: {exc!r}")
            self._send_json(502, {"error": f"PD handoff failed: {exc}"})

    def _handoff(self, suffix: str, body: dict) -> None:
        # ---- stage 1: prefill (1 token, no stream) ----------------------
        p_body = dict(body)
        p_body["max_tokens"] = 1
        if "max_completion_tokens" in p_body:
            p_body["max_completion_tokens"] = 1
        p_body["stream"] = False
        p_body.pop("stream_options", None)
        p_body["kv_transfer_params"] = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
        }

        status, payload = post(
            f"{self.server.prefill}{suffix}", p_body, self.server.timeout
        )
        if not 200 <= status < 300:
            raise RuntimeError(f"prefill returned HTTP {status}: {payload[:300]!r}")

        try:
            exported = json.loads(payload).get("kv_transfer_params")
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"prefill response is not JSON: {payload[:200]!r}") from exc
        if not isinstance(exported, dict):
            raise RuntimeError(
                "prefill response carries no kv_transfer_params — is the prefill "
                "engine started with --kv-transfer-config kv_role=kv_producer?"
            )

        # ---- stage 2: decode (relay the answer) -------------------------
        d_body = dict(body)
        d_body["kv_transfer_params"] = exported
        stream = bool(body.get("stream"))

        if stream:
            self._relay_stream(f"{self.server.decode}{suffix}", d_body)
        else:
            status, payload = post(
                f"{self.server.decode}{suffix}", d_body, self.server.timeout
            )
            if not 200 <= status < 300:
                raise RuntimeError(f"decode returned HTTP {status}: {payload[:300]!r}")
            self._send(status, payload, "application/json")

        with LOCK:
            STATS["ok"] += 1

    def _relay_stream(self, url: str, body: dict) -> None:
        data = json.dumps(body).encode("utf-8")
        request = urlrequest.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            response = OPENER.open(request, timeout=self.server.timeout)
        except urlerror.HTTPError as exc:
            raise RuntimeError(f"decode returned HTTP {exc.code}: {exc.read()[:300]!r}") from exc

        with response:
            self.send_response(response.status)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                chunk = response.read(1024)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        self.close_connection = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9060)
    parser.add_argument("--prefill", required=True, help="http://host:port of the P engine")
    parser.add_argument("--decode", required=True, help="http://host:port of the D engine")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.prefill = args.prefill.rstrip("/")
    server.decode = args.decode.rstrip("/")
    server.timeout = args.timeout
    log(f"listening on http://{args.host}:{args.port}  prefill={server.prefill} decode={server.decode}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
