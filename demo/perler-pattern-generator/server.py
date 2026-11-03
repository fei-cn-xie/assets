# -*- coding: utf-8 -*-
"""server.py — 拼豆图纸生成器本地服务器（仅标准库 + Pillow）。

用法：python server.py [端口]   然后浏览器打开 http://127.0.0.1:8765
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

import core

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, "index.html")
PALETTE_PATH = os.path.join(HERE, "palette.json")

MAX_PIXELS = 4000 * 4000


def _json_bytes(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "PerlerGen/1.0"

    # ---- helpers

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, code: int = 200):
        self._send(code, _json_bytes(obj), "application/json; charset=utf-8")

    def log_message(self, fmt, *args):
        pass  # 安静模式

    # ---- GET

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(INDEX, "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif self.path == "/api/palette":
            with open(PALETTE_PATH, "rb") as f:
                self._send(200, f.read(), "application/json; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    # ---- POST /api/convert

    def do_POST(self):
        if self.path != "/api/convert":
            self._send_json({"error": "unknown endpoint"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 64 * 1024 * 1024:
                raise ValueError("bad request size")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))

            data_url = payload["image"]
            b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
            raw = base64.b64decode(b64)
            img = Image.open(io.BytesIO(raw))
            img.load()
            if img.width * img.height > MAX_PIXELS:
                img.thumbnail((3000, 3000), Image.LANCZOS)

            width = int(payload.get("width", 29))
            height = int(payload.get("height", 29))
            if not (4 <= width <= 400 and 4 <= height <= 400):
                raise ValueError("width/height 需在 4~400 之间")

            palette = core.load_palette(PALETTE_PATH)

            pattern = core.convert(
                img, width, height, palette,
                dither=bool(payload.get("dither", False)),
                brightness=float(payload.get("brightness", 1.0)),
                saturation=float(payload.get("saturation", 1.0)),
                contrast=float(payload.get("contrast", 1.0)),
            )

            cell = int(payload.get("cell", 18))
            labels = bool(payload.get("labels", True))
            symbol_mode = bool(payload.get("symbols", False))

            sheet = core.render(pattern, cell=cell, labels=labels, symbol_mode=symbol_mode)
            flat = core.render_flat(pattern, cell=10)

            def to_png(im: Image.Image) -> str:
                buf = io.BytesIO()
                im.save(buf, "PNG")
                return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

            counts = [
                {"code": b.code, "name": b.name, "brand_code": b.brand_code, "count": n}
                for b, n in pattern.counts()
            ]
            self._send_json({
                "width": width,
                "height": height,
                "total": pattern.total_beads(),
                "colors": len(counts),
                "boards": pattern.boards(),
                "sheet_png": to_png(sheet),
                "flat_png": to_png(flat),
                "counts": counts,
                "csv": core.counts_csv(pattern),
            })
        except Exception as e:  # noqa: BLE001
            self._send_json({"error": f"{type(e).__name__}: {e}"}, 400)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    url = f"http://127.0.0.1:{port}"
    print(f"拼豆图纸生成器已启动：{url}  （关闭本窗口即可退出）")
    if "--no-browser" not in sys.argv:
        import threading
        import webbrowser

        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\n已退出。")


if __name__ == "__main__":
    main()
