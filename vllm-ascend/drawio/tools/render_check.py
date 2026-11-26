"""Self-check: parse a .drawio file and render it to PNG/SVG ourselves.

Why this exists: the diagram is written as explicit geometry (every vertex has
x/y/w/h, every edge has endpoints), so rendering the *file on disk* is a real
round-trip check -- if the parsed geometry looks right, draw.io shows it right.

On top of drawing, it asserts draw.io's visibility rule: a cell is only painted
when its parent chain reaches cell "1" (the default layer). A cell whose parent
is missing or unknown is attached to the root cell "0" and is INVISIBLE in
draw.io -- that exact bug shipped once (vertices missing parent="1"), and this
tool exits non-zero when it happens again.

Usage:
    python render_check.py diagram.drawio [-o outdir] [--scale 1.0]
                                          [--only 0,2] [--svg]
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys

try:                       # Windows 控制台默认 GBK；幂等，避免 import 链重复包装
    import io as _io
    if "utf-8" not in (getattr(sys.stdout, "encoding", "") or "").lower():
        sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                      errors="replace")
except Exception:
    pass

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFont

FONT = r"C:\Windows\Fonts\msyh.ttc"
FONT_B = r"C:\Windows\Fonts\msyhbd.ttc"
MONO = r"C:\Windows\Fonts\consola.ttf"
TAG = re.compile(r"<[^>]+>")


# ------------------------------------------------------------------ styles ---
def parse_style(style: str) -> dict[str, str]:
    out: dict[str, str] = {}
    tokens = (style or "").split(";")
    for i, tok in enumerate(tokens):
        if not tok:
            continue
        if "=" in tok:
            k, _, v = tok.partition("=")
            out[k.strip()] = v.strip()
        elif i == 0:
            out["__shape__"] = tok.strip()
    return out


def unescape(text: str) -> str:
    return (text.replace("&lt;", "<").replace("&gt;", ">")
            .replace("&quot;", '"').replace("&#10;", "\n").replace("&amp;", "&"))


# ------------------------------------------------------------------- model ---
@dataclass
class V:
    id: str
    label: str
    style: dict[str, str]
    x: float
    y: float
    w: float
    h: float
    parent: str | None


@dataclass
class E:
    id: str
    label: str
    style: dict[str, str]
    parent: str | None
    source: str | None
    target: str | None
    src_pt: tuple[float, float] | None
    dst_pt: tuple[float, float] | None
    points: list[tuple[float, float]] = field(default_factory=list)


def load(path: str):
    raw = open(path, "rb").read()
    if raw[:3] == b"\xef\xbb\xbf":
        raw = raw[3:]
    root = ET.fromstring(raw.decode("utf-8"))
    pages = []
    for diag in root.findall("diagram"):
        model = diag.find("mxGraphModel")
        r = model.find("root")
        verts: dict[str, V] = {}
        edges: dict[str, E] = {}
        parents: dict[str, str | None] = {}
        for cell in r.findall("mxCell"):
            cid = cell.get("id") or ""
            parents[cid] = cell.get("parent")
            geo = cell.find("mxGeometry")
            if cell.get("edge") == "1":
                pts: list[tuple[float, float]] = []
                src_pt = dst_pt = None
                if geo is not None:
                    arr = geo.find("Array")
                    if arr is not None:
                        pts = [(float(p.get("x", 0)), float(p.get("y", 0)))
                               for p in arr.findall("mxPoint")]
                    for p in geo.findall("mxPoint"):
                        kind = p.get("as")
                        xy = (float(p.get("x", 0)), float(p.get("y", 0)))
                        if kind == "sourcePoint":
                            src_pt = xy
                        elif kind == "targetPoint":
                            dst_pt = xy
                edges[cid] = E(cid, unescape(cell.get("value") or ""),
                               parse_style(cell.get("style") or ""),
                               cell.get("parent"), cell.get("source"),
                               cell.get("target"), src_pt, dst_pt, pts)
            else:
                if geo is None:
                    continue
                verts[cid] = V(cid, unescape(cell.get("value") or ""),
                               parse_style(cell.get("style") or ""),
                               float(geo.get("x", 0) or 0), float(geo.get("y", 0) or 0),
                               float(geo.get("width", 0) or 0),
                               float(geo.get("height", 0) or 0), cell.get("parent"))
        pages.append({
            "name": diag.get("name"),
            "w": int(float(model.get("pageWidth") or 1000)),
            "h": int(float(model.get("pageHeight") or 800)),
            "verts": verts, "edges": edges, "parents": parents,
        })
    return pages


def visible(cid: str, parents: dict[str, str | None]) -> bool:
    """draw.io paints only cells whose ancestor chain reaches layer cell '1'."""
    seen = set()
    cur = parents.get(cid)
    while cur is not None and cur not in seen:
        if cur == "1":
            return True
        if cur == "0":
            return False
        seen.add(cur)
        cur = parents.get(cur)
    return False


# ---------------------------------------------------------------- painting ---
def font(size: float, bold=False, mono=False):
    path = MONO if mono else (FONT_B if bold else FONT)
    try:
        return ImageFont.truetype(path, max(6, int(round(size))))
    except OSError:
        return ImageFont.load_default()


def wrap(text: str, fnt, max_w: float, d) -> list[str]:
    if not text:
        return [""]
    out, cur = [], ""
    for ch in text:
        if d.textlength(cur + ch, font=fnt) > max_w and cur:
            out.append(cur)
            cur = ch
        else:
            cur += ch
    out.append(cur)
    return out


def draw_text(d, v: V, scale: float) -> bool:
    """Draw the label; returns True when it overflows the box.

    Wrapping and the overflow verdict are computed at TRUE size (scale 1), so the
    result does not depend on --scale. Drawing alone is scaled.
    """
    st = v.style
    base = float(st.get("fontSize", 11) or 11)
    bold = st.get("fontStyle") == "1"
    mono = "Courier" in (st.get("fontFamily") or "")
    spans = []
    for raw in v.label.split("<br>"):
        color = st.get("fontColor") or "#333333"
        m = re.search(r'color="(#[0-9a-fA-F]{6})"', raw)
        if m:
            color = m.group(1)
        fsz = re.search(r"font-size:(\d+)px", raw)
        size = float(fsz.group(1)) if fsz else base
        plain = TAG.sub("", raw)
        spans.append((plain, size, color, bool(fsz)))
    pad = 7 if st.get("spacingLeft") == "7" else (
        4 if st.get("spacingLeft") == "4" else 4)
    avail = max(10.0, v.w - pad * 2)          # 真实尺寸
    meas = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    lines: list[tuple[str, float, str, bool]] = []
    for plain, size, color, small in spans:
        f = font(size, bold=bold and not small, mono=mono)
        for seg in wrap(plain, f, avail, meas):
            lines.append((seg, size, color, small))
    total = sum(size * 1.32 for _, size, _, _ in lines)
    overflow = total > v.h - 2

    top = v.y * scale + 3 * scale
    if st.get("verticalAlign") != "top":
        free = (v.h - total) * scale
        if free > 2:
            top += free / 2
    cy, center = top, st.get("align") == "center"
    for seg, size, color, small in lines:
        f = font(max(7, size * scale), bold=bold and not small, mono=mono)
        wpx = d.textlength(seg, font=f)
        tx = ((v.x + v.w / 2) * scale - wpx / 2) if center else (v.x + pad) * scale
        d.text((tx, cy), seg, fill=color, font=f)
        cy += size * 1.32 * scale
    return overflow


def shape_path(d, v: V, scale: float):
    st = v.style
    box = [v.x * scale, v.y * scale, (v.x + v.w) * scale, (v.y + v.h) * scale]
    fill = st.get("fillColor", "#ffffff")
    stroke = st.get("strokeColor", "#000000")
    fill = fill if fill and fill.startswith("#") else "#ffffff"
    if fill == "none":
        fill = None
    stroke = stroke if stroke and stroke.startswith("#") else "#666666"
    kind = st.get("__shape__", "")
    dashed = st.get("dashed") == "1"
    width = max(1, int(round(float(st.get("strokeWidth", 1) or 1))))
    if kind == "text":
        return
    if kind == "ellipse":
        d.ellipse(box, fill=fill, outline=stroke, width=width)
    elif kind == "shape" and st.get("shape") == "note":
        c = min(12 * scale, v.h * scale * 0.4)
        d.polygon([(box[0], box[1]), (box[2] - c, box[1]), (box[2], box[1] + c),
                   (box[2], box[3]), (box[0], box[3])], fill=fill, outline=stroke)
    elif kind == "shape" and st.get("shape") == "cylinder3":
        d.rounded_rectangle(box, radius=6 * scale, fill=fill, outline=stroke)
    else:
        radius = 4 * scale if st.get("rounded") == "1" else 0
        if dashed:
            d.rectangle(box, outline=stroke, width=width)
        else:
            d.rounded_rectangle(box, radius=radius, fill=fill, outline=stroke,
                                width=width)


def anchor(v: V, style: dict[str, str], which: str) -> tuple[float, float]:
    kx, ky = ("exitX", "exitY") if which == "exit" else ("entryX", "entryY")
    if style.get(kx) and style.get(ky):
        return (v.x + v.w * float(style[kx]), v.y + v.h * float(style[ky]))
    return (v.x + v.w / 2, v.y + v.h / 2)


def draw_edge(d, e: E, verts: dict[str, V], scale: float) -> bool:
    """Returns True when an endpoint cannot be resolved."""
    bad = False
    pts = []
    if e.source and e.source in verts:
        pts.append(anchor(verts[e.source], e.style, "exit"))
    elif e.src_pt:
        pts.append(e.src_pt)
    else:
        bad = True
    pts.extend(e.points)
    if e.target and e.target in verts:
        pts.append(anchor(verts[e.target], e.style, "entry"))
    elif e.dst_pt:
        pts.append(e.dst_pt)
    else:
        bad = True
    if len(pts) < 2:
        return bad
    color = e.style.get("strokeColor", "#000000")
    color = color if color.startswith("#") else "#333333"
    dashed = e.style.get("dashed") == "1"
    width = max(1, int(round(float(e.style.get("strokeWidth", 1) or 1))))
    sp = [(p[0] * scale, p[1] * scale) for p in pts]
    for a, b in zip(sp, sp[1:]):
        if dashed:
            total = math.hypot(b[0] - a[0], b[1] - a[1])
            n = max(1, int(total // 7))
            for i in range(0, n, 2):
                t0, t1 = i / n, min(1.0, (i + 1) / n)
                d.line([(a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0),
                        (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)],
                       fill=color, width=width)
        else:
            d.line([a, b], fill=color, width=width)
    if e.style.get("endArrow", "classic") != "none":
        a, b = sp[-2], sp[-1]
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        s = 7 * scale
        for off in (0.45, -0.45):
            d.line([(b[0] - s * math.cos(ang - off), b[1] - s * math.sin(ang - off)),
                    b], fill=color, width=width)
    if e.label:
        mx = (sp[0][0] + sp[-1][0]) / 2
        my = (sp[0][1] + sp[-1][1]) / 2
        f = font(9 * scale)
        wpx = d.textlength(e.label, font=f)
        d.rectangle([mx - wpx / 2 - 2, my - 13 * scale, mx + wpx / 2 + 2,
                     my - 2 * scale], fill="#ffffff")
        d.text((mx - wpx / 2, my - 13 * scale), e.label, fill="#222222", font=f)
    return bad


def to_svg(page) -> str:
    """Vector export of the same parsed geometry (for zooming / diffing)."""
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{page["w"]}" '
           f'height="{page["h"]}" viewBox="0 0 {page["w"]} {page["h"]}">',
           f'<rect width="{page["w"]}" height="{page["h"]}" fill="#ffffff"/>']
    par = page["parents"]
    for v in page["verts"].values():
        if not visible(v.id, par):
            continue
        st = v.style
        fill = st.get("fillColor", "#ffffff") or "#ffffff"
        stroke = st.get("strokeColor", "#000000") or "#666666"
        dash = ' stroke-dasharray="6 4"' if st.get("dashed") == "1" else ""
        kind = st.get("__shape__", "")
        if kind == "text":
            out.append(f'<text x="{v.x}" y="{v.y + 14}" font-size="11" '
                       f'fill="{st.get("fontColor", "#333333")}">'
                       f'{_xml(TAG.sub("", v.label))}</text>')
            continue
        rx = 4 if st.get("rounded") == "1" else 0
        if kind == "ellipse":
            out.append(f'<ellipse cx="{v.x + v.w / 2}" cy="{v.y + v.h / 2}" '
                       f'rx="{v.w / 2}" ry="{v.h / 2}" fill="{fill}" '
                       f'stroke="{stroke}"{dash}/>')
        else:
            out.append(f'<rect x="{v.x}" y="{v.y}" width="{v.w}" height="{v.h}" '
                       f'rx="{rx}" fill="{fill}" stroke="{stroke}"{dash}/>')
        ty = v.y + 14
        for line in v.label.split("<br>"):
            plain = TAG.sub("", line)
            fs = re.search(r"font-size:(\d+)px", line)
            size = int(fs.group(1)) if fs else int(float(st.get("fontSize", 11)))
            col = st.get("fontColor", "#333333")
            m = re.search(r'color="(#[0-9a-fA-F]{6})"', line)
            if m:
                col = m.group(1)
            out.append(f'<text x="{v.x + 6}" y="{ty}" font-size="{size}" '
                       f'fill="{col}" font-family="Microsoft YaHei">{_xml(plain)}</text>')
            ty += size * 1.35
    for e in page["edges"].values():
        if not visible(e.id, par):
            continue
        pts = []
        if e.source and e.source in page["verts"]:
            pts.append(anchor(page["verts"][e.source], e.style, "exit"))
        elif e.src_pt:
            pts.append(e.src_pt)
        pts.extend(e.points)
        if e.target and e.target in page["verts"]:
            pts.append(anchor(page["verts"][e.target], e.style, "entry"))
        elif e.dst_pt:
            pts.append(e.dst_pt)
        if len(pts) < 2:
            continue
        d = " ".join(f"{'M' if i == 0 else 'L'}{x},{y}"
                     for i, (x, y) in enumerate(pts))
        col = e.style.get("strokeColor", "#333333") or "#333333"
        dash = ' stroke-dasharray="5 4"' if e.style.get("dashed") == "1" else ""
        out.append(f'<path d="{d}" fill="none" stroke="{col}" '
                   f'marker-end="url(#a)"{dash}/>')
    out.append('<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" '
               'markerWidth="6" markerHeight="6" orient="auto">'
               '<path d="M0,0 L10,5 L0,10 z" fill="#333333"/></marker></defs>')
    out.append("</svg>")
    return "\n".join(out)


def _xml(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


# --------------------------------------------------------------------- main ---
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("-o", "--outdir", default=None)
    ap.add_argument("--scale", type=float, default=0.9)
    ap.add_argument("--only", default=None, help="comma separated page indexes")
    ap.add_argument("--svg", action="store_true")
    args = ap.parse_args()

    pages = load(args.path)
    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(args.path)),
                                        "render")
    os.makedirs(outdir, exist_ok=True)
    only = {int(x) for x in args.only.split(",")} if args.only else None

    problems: list[str] = []
    for i, page in enumerate(pages):
        par = page["parents"]
        hidden_v = [v.id for v in page["verts"].values() if not visible(v.id, par)]
        hidden_e = [e.id for e in page["edges"].values() if not visible(e.id, par)]
        if hidden_v or hidden_e:
            problems.append(f"[{i} {page['name']}] invisible in draw.io: "
                            f"{len(hidden_v)} shapes, {len(hidden_e)} edges")
        if only is not None and i not in only:
            continue
        img = Image.new("RGB", (max(10, int(page["w"] * args.scale)),
                                max(10, int(page["h"] * args.scale))), "#ffffff")
        d = ImageDraw.Draw(img)
        for v in page["verts"].values():
            if not visible(v.id, par) or v.w <= 0 or v.h <= 0:
                continue
            shape_path(d, v, args.scale)
            if v.label and v.style.get("__shape__") != "text":
                if draw_text(d, v, args.scale):
                    problems.append(f"[{i} {page['name']}] text overflows box {v.id}")
            elif v.label:
                draw_text(d, v, args.scale)
        for e in page["edges"].values():
            if not visible(e.id, par):
                continue
            if draw_edge(d, e, page["verts"], args.scale):
                problems.append(f"[{i} {page['name']}] edge {e.id} has an "
                                f"unresolved endpoint")
        png = os.path.join(outdir, f"page{i:02d}.png")
        img.save(png)
        svg = os.path.join(outdir, f"page{i:02d}.svg")
        if args.svg:
            with open(svg, "w", encoding="utf-8") as f:
                f.write(to_svg(page))
        print(f"[{i:02d}] {page['name']}: {page['w']}x{page['h']} "
              f"shapes={len(page['verts'])} edges={len(page['edges'])} -> {png}")

    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems[:30]:
            print("  -", p)
        return 1
    print("\nOK: every cell is visible in draw.io, no text overflow, "
          "no dangling edge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
