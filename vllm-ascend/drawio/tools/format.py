"""Pretty-print a one-line draw.io export into reviewable multi-line XML.

draw.io itself saves pretty XML; the MCP exporter writes a single line, which is
unreviewable in a diff. This only adds whitespace between tags, so the document
stays byte-equivalent in meaning.
"""
from __future__ import annotations

import re
import sys


def pretty(src: str, indent: str = "  ") -> str:
    # one tag per line: any "><" boundary becomes a line break
    body = re.sub(r">\s*<", "><", src.strip())
    parts = body.split("<")
    lines: list[str] = []
    depth = 0
    for i, chunk in enumerate(parts):
        if not chunk:
            continue
        tag = "<" + chunk
        closing = chunk.startswith("/")
        self_closing = chunk.rstrip().endswith("/>") or chunk.startswith("?")
        if closing:
            depth = max(0, depth - 1)
        lines.append(indent * depth + tag)
        if not closing and not self_closing and not chunk.startswith("!"):
            depth += 1
    return "\n".join(lines) + "\n"


def normalized(src: str, indent: str = "  ") -> str:
    """Reproducible on-disk form: XML declaration + one tag per line + LF."""
    body = src.lstrip()
    if body.startswith("<?xml"):
        body = body[body.index("?>") + 2:]
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + pretty(body, indent)


def main() -> int:
    path = sys.argv[1]
    raw = open(path, "rb").read()
    src = raw.decode("utf-8")
    out = normalized(src)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(out)
    print(f"{path}: {len(raw)} bytes -> {len(out.encode('utf-8'))} bytes, "
          f"{out.count(chr(10))} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
