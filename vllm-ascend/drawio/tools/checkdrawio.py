"""Structural validator for the generated .drawio file.

Checks the invariants draw.io itself always writes, so a file that passes here
renders in the desktop app (a missing parent= silently hides nearly every cell).
"""
from __future__ import annotations

import sys

try:                       # Windows 控制台默认 GBK；幂等，避免 import 链重复包装
    import io as _io
    if "utf-8" not in (getattr(sys.stdout, "encoding", "") or "").lower():
        sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                      errors="replace")
except Exception:
    pass

import xml.etree.ElementTree as ET

REQUIRED = {
    "shapes": ("id", "value", "style", "vertex", "parent"),
    "edges": ("id", "style", "edge", "parent"),
}


def check(path: str) -> int:
    problems: list[str] = []
    raw = open(path, "rb").read()
    if raw[:3] == b"\xef\xbb\xbf":
        problems.append("file starts with a UTF-8 BOM")
    text = raw.decode("utf-8")
    if not text.lstrip().startswith("<?xml"):
        problems.append("missing XML declaration")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        print("XML NOT WELL-FORMED:", exc)
        return 1

    pages = root.findall("diagram")
    if not pages:
        problems.append("no <diagram> pages")
    for page in pages:
        name = page.get("name")
        model = page.find("mxGraphModel")
        if model is None:
            problems.append(f"[{name}] no mxGraphModel")
            continue
        root_cell = model.find("root")
        if root_cell is None:
            problems.append(f"[{name}] no <root>")
            continue
        ids = set()
        shapes = edges = 0
        for cell in root_cell.iter("mxCell"):
            cid = cell.get("id")
            if cid in ids:
                problems.append(f"[{name}] duplicate id {cid}")
            ids.add(cid)
            if cid in ("0", "1"):
                continue
            if cell.get("edge") == "1":
                edges += 1
                kind = "edges"
            else:
                shapes += 1
                kind = "shapes"
                if cell.find("mxGeometry") is None:
                    problems.append(f"[{name}] shape {cid} has no mxGeometry")
            for attr in REQUIRED[kind]:
                if cell.get(attr) is None:
                    problems.append(f"[{name}] {kind[:-1]} {cid} missing {attr}=")
            if cell.get("parent") not in ids and cell.get("parent") != "1":
                problems.append(f"[{name}] cell {cid} points at unknown "
                                f"parent={cell.get('parent')}")
            for other in cell.findall("mxCell"):
                problems.append(f"[{name}] nested mxCell inside {cid}")
        print(f"[{name}] shapes={shapes} edges={edges} "
              f"page={model.get('pageWidth')}x{model.get('pageHeight')}")

    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems[:40]:
            print("  -", p)
        return 1
    print("\nOK: structure matches draw.io's own save format")
    return 0


if __name__ == "__main__":
    raise SystemExit(check(sys.argv[1]))
