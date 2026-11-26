"""Tiny multi-page draw.io authoring kit: deterministic layout, no hand-placed XML.

Usage:
    doc = Doc()
    p = doc.page("页面名", pw=2000)
    p.box(40, 40, 200, 60, "标题", "box_info")
    doc.save("out.drawio")
"""
from __future__ import annotations

import html
import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------- palette ----
# ONE colour = ONE concept, same on every page. Two axes only:
#   fill  = 角色/归属      (client / edge / cloud-P / cloud-D / PD Proxy / 中立)
#   line  = 平面           (控制面 ZMQ / 数据面 UCX·Mooncake / HTTP)
# 状态类只用一种色：失败/取消/超时 = fail。就绪/完成不另设色，靠文字表达。
C = {
    "client": ("#f5f5f5", "#666666"),     # 客户端
    "edge": ("#dae8fc", "#6c8ebf"),       # 边端 edge 实例
    "cloudp": ("#ffe6cc", "#d79b00"),     # 云 P（prefill）
    "cloudd": ("#d5e8d4", "#82b366"),     # 云 D（decode）
    "proxy": ("#b0e3e6", "#0e8088"),      # PD Proxy
    "neutral": ("#ffffff", "#b3b3b3"),    # 中立：通用 vLLM 代码/进程骨架/分组带/表格
    "fail": ("#f8cecc", "#b85450"),       # 失败 / 取消 / 超时 / 终止
    # 进程泳道灰度梯度：只用于"按进程切泳道"的页面（P02/P05/P09）。
    # 灰阶不是颜色，所以不与角色色冲突；图例里单列一项说明。
    "proc1": ("#ffffff", "#9e9e9e"),
    "proc2": ("#f1f1f1", "#9e9e9e"),
    "proc3": ("#e4e4e4", "#8a8a8a"),
    "proc4": ("#d7d7d7", "#8a8a8a"),
    "proc5": ("#c9c9c9", "#757575"),
    # 平面（只用于线，不用于填充）
    "ctrl": ("#ffffff", "#9673a6"),       # 控制面 ZMQ
    "data": ("#fff2cc", "#d6b656"),       # 数据面 UCX / Mooncake
    "http": ("#ffffff", "#666666"),       # HTTP
    # 兼容旧名
    "grey": ("#ffffff", "#b3b3b3"),
    "code": ("#f5f5f5", "#999999"),
    "ok": ("#ffffff", "#82b366"),
    "warn": ("#f8cecc", "#b85450"),
    "phase": ("#ffffff", "#b3b3b3"),
}

# 统一的 legend 项（顺序固定，措辞固定）
LEGEND_ITEMS = [
    ("客户端", "client"), ("边端 edge", "edge"), ("云 P (prefill)", "cloudp"),
    ("云 D (decode)", "cloudd"), ("PD Proxy", "proxy"), ("中立/通用", "neutral"),
    ("失败/取消", "fail"),
]
# 图例色块的期望填色（供 color_audit.py 校验）
LEGEND_ITEMS_ROLES = [(C[role][0], role) for _, role in LEGEND_ITEMS]
LEGEND_LINES = [
    ("控制面 ZMQ", "ctrl"), ("数据面 UCX·Mooncake", "data"), ("HTTP", "http"),
]


def style(name: str, extra: str = "") -> str:
    """Named style table. Kept flat so every page reuses the same look."""
    return STYLES[name] + extra


STYLES = {
    # ---- boxes -------------------------------------------------------------
    "box": "rounded=1;html=1;whiteSpace=wrap;align=left;verticalAlign=middle;"
           "spacingLeft=7;spacingRight=5;fontSize=11;",
    "box_center": "rounded=1;html=1;whiteSpace=wrap;align=center;"
                  "verticalAlign=middle;fontSize=11;",
    "box_bold": "rounded=1;html=1;whiteSpace=wrap;align=left;verticalAlign=middle;"
                "spacingLeft=7;fontSize=11;fontStyle=1;",
    "head": "rounded=0;html=1;whiteSpace=wrap;align=center;verticalAlign=middle;"
            "fontSize=12;fontStyle=1;",
    "phase": "rounded=0;html=1;whiteSpace=wrap;align=left;verticalAlign=middle;"
             "spacingLeft=0;fontSize=11;fontStyle=1;dashed=1;",
    "lane": "rounded=0;html=1;whiteSpace=wrap;align=center;verticalAlign=top;"
            "fontSize=12;fontStyle=1;opacity=45;",
    "note": "shape=note;html=1;whiteSpace=wrap;align=left;verticalAlign=top;"
            "spacingLeft=6;spacingTop=4;fontSize=10;size=11;",
    "code": "rounded=0;html=1;whiteSpace=wrap;align=left;verticalAlign=middle;"
            "spacingLeft=4;fontFamily=Courier New;fontSize=9;",
    "text": "text;html=1;whiteSpace=wrap;align=left;verticalAlign=middle;fontSize=11;",
    "text_b": "text;html=1;whiteSpace=wrap;align=left;verticalAlign=middle;"
              "fontSize=12;fontStyle=1;",
    "title": "text;html=1;whiteSpace=wrap;align=left;verticalAlign=middle;"
             "fontSize=17;fontStyle=1;",
    "subtitle": "text;html=1;whiteSpace=wrap;align=left;verticalAlign=middle;"
                "fontSize=11;fontColor=#666666;",
    "ellipse": "ellipse;html=1;whiteSpace=wrap;align=center;verticalAlign=middle;"
               "fontSize=11;",
    "rhombus": "rhombus;html=1;whiteSpace=wrap;align=center;verticalAlign=middle;"
               "fontSize=11;",
    "cylinder": "shape=cylinder3;html=1;whiteSpace=wrap;align=center;"
                "verticalAlign=middle;fontSize=11;boundedLbl=1;size=8;",
    "process": "shape=process;html=1;whiteSpace=wrap;align=center;"
               "verticalAlign=middle;fontSize=11;",
    "db": "shape=cylinder3;html=1;whiteSpace=wrap;align=center;verticalAlign=middle;"
          "fontSize=11;direction=south;",
    # ---- edges -------------------------------------------------------------
    "arrow": "endArrow=block;html=1;rounded=0;fontSize=10;",
    "arrow_open": "endArrow=open;html=1;rounded=0;fontSize=10;",
    "arrow_dash": "endArrow=block;html=1;rounded=0;dashed=1;fontSize=10;",
    "arrow_dash_open": "endArrow=open;html=1;rounded=0;dashed=1;fontSize=10;",
    "arrow_thick": "endArrow=block;html=1;rounded=0;fontSize=10;strokeWidth=2;",
    "line": "endArrow=none;html=1;rounded=0;",
    "lifeline": "endArrow=none;html=1;rounded=0;dashed=1;strokeColor=#b3b3b3;",
    "bracket": "endArrow=none;html=1;rounded=0;strokeColor=#999999;",
}


# ------------------------------------------------------------------- cells ---
@dataclass
class Cell:
    id: str
    kind: str            # "v" vertex, "e" edge, "t" text
    value: str
    style: str
    x: float = 0
    y: float = 0
    w: float = 0
    h: float = 0
    source: str | None = None
    target: str | None = None
    points: list[tuple[float, float]] = field(default_factory=list)
    src_pt: tuple[float, float] | None = None
    dst_pt: tuple[float, float] | None = None
    parent: str | None = None
    decor: bool = False


def esc(v: str) -> str:
    return html.escape(v, quote=False).replace("\n", "")


def _attr(v: str) -> str:
    return (v.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;").replace("\n", "&#10;"))


class Page:
    def __init__(self, doc: "Doc", name: str, pw: int = 2200, ph: int = 1500,
                 grid: int = 10):
        self.doc = doc
        self.name = name
        self.pw, self.ph = pw, ph
        self.grid = grid
        self.cells: list[Cell] = []
        self.pending: list = []      # builders that emit lazily (see Doc.xml)
        self.fill_map: dict[str, str] = {}   # page-local alias -> canonical fill role
        self._n = 0

    def role_of(self, role: str | None) -> str | None:
        if role is None:
            return None
        return self.fill_map.get(role, role)

    # -- id ------------------------------------------------------------------
    def nid(self, prefix: str = "n") -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    # -- primitives ----------------------------------------------------------
    def box(self, x, y, w, h, label, st="box", role: str | None = None,
            bold_top: bool = False, cid: str | None = None, parent=None,
            decor: bool = False, extra_style: str = "") -> str:
        s = STYLES[st]
        role = self.role_of(role)
        if role:
            fill, stroke = C[role]
            s += f"fillColor={fill};strokeColor={stroke};"
        s += extra_style
        if bold_top and label:
            first, _, rest = label.partition("<br>")
            label = f"<b>{first}</b>" + (f"<br>{rest}" if rest else "")
        cid = cid or self.nid()
        c = Cell(cid, "v", label, s, x, y, w, h, parent=parent)
        c.decor = decor or st in ("lane", "phase", "text", "title", "subtitle",
                                  "text_b", "lifeline", "code")
        self.cells.append(c)
        return cid

    def text(self, x, y, w, h, label, st="text", role=None) -> str:
        cid = self.nid("t")
        s = STYLES[st]
        role = self.role_of(role)
        if role:
            fill, stroke = C[role]
            s += f"fillColor={fill};strokeColor={stroke};"
        c = Cell(cid, "v", label, s, x, y, w, h)
        c.decor = True          # 纯文字注记，不参与"内容盒"的重叠判断
        self.cells.append(c)
        return cid

    def edge(self, src: str | None = None, dst: str | None = None, label: str = "",
             st="arrow", role: str | None = None, points=None,
             src_pt=None, dst_pt=None, cid: str | None = None,
             label_pos: float | None = None, decor: bool = False,
             extra_style: str = "") -> str:
        s = STYLES[st]
        if role:
            _, stroke = C[role]
            s += f"strokeColor={stroke};"
            if st in ("arrow", "arrow_thick"):
                s += f"fillColor={stroke};"
            if role in ("ctrl", "data", "http"):
                s += "fontColor=#333333;"
        if label_pos is not None:
            s += f"labelBackgroundColor=#ffffff;"
        s += extra_style
        cid = cid or self.nid("e")
        c = Cell(cid, "e", label, s,
                 source=src, target=dst,
                 points=list(points or []),
                 src_pt=src_pt, dst_pt=dst_pt)
        c.decor = decor
        self.cells.append(c)
        return cid

    # -- convenience ---------------------------------------------------------
    def title_block(self, title: str, subtitle: str = "", x=20, y=14, w=1400):
        self.text(x, y, w, 26, title, "title")
        if subtitle:
            self.text(x, y + 28, w, 20, subtitle, "subtitle")

    def save(self):
        return

    # -- emit ----------------------------------------------------------------
    def _cell_xml(self, c: Cell) -> str:
        common = f'id="{c.id}" value="{_attr(c.value)}" style="{_attr(c.style)}"'
        # draw.io always writes parent= on every cell; without it the cell is
        # attached to the root ("0") and never rendered.
        parent = c.parent if c.parent else "1"
        if c.kind == "e":
            attrs = [common, 'edge="1"', f'parent="{parent}"']
            if c.source:
                attrs.append(f'source="{c.source}"')
            if c.target:
                attrs.append(f'target="{c.target}"')
            geo = '<mxGeometry relative="1" as="geometry">'
            inner = ""
            if c.src_pt:
                inner += (f'<mxPoint x="{_num(c.src_pt[0])}" '
                          f'y="{_num(c.src_pt[1])}" as="sourcePoint"/>')
            if c.dst_pt:
                inner += (f'<mxPoint x="{_num(c.dst_pt[0])}" '
                          f'y="{_num(c.dst_pt[1])}" as="targetPoint"/>')
            if c.points:
                inner += '<Array as="points">' + "".join(
                    f'<mxPoint x="{_num(x)}" y="{_num(y)}"/>' for x, y in c.points
                ) + "</Array>"
            geo += inner + "</mxGeometry>"
            return f"<mxCell {' '.join(attrs)}>{geo}</mxCell>"
        attrs = [common, 'vertex="1"', f'parent="{parent}"']
        geo = (f'<mxGeometry x="{_num(c.x)}" y="{_num(c.y)}" '
               f'width="{_num(c.w)}" height="{_num(c.h)}" as="geometry"/>')
        return f"<mxCell {' '.join(attrs)}>{geo}</mxCell>"

    def xml(self, page_id: str) -> str:
        head = (f'<mxGraphModel dx="1600" dy="1000" grid="1" gridSize="{self.grid}" '
                f'guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" '
                f'pageScale="1" pageWidth="{self.pw}" pageHeight="{self.ph}" '
                f'math="0" shadow="0">')
        body = "".join(self._cell_xml(c) for c in self.cells)
        root = f'<root><mxCell id="0"/><mxCell id="1" parent="0"/>{body}</root>'
        return (f'<diagram id="{page_id}" name="{_attr(self.name)}">{head}{root}'
                f'</mxGraphModel></diagram>')


def _num(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.1f}"


# -------------------------------------------------------------- validation ---
def _rect_overlap(a: Cell, b: Cell, pad: float = 0) -> bool:
    return not (a.x + a.w <= b.x + pad or b.x + b.w <= a.x + pad
                or a.y + a.h <= b.y + pad or b.y + b.h <= a.y + pad)


def _seg_hits_rect(p1, p2, r: Cell, pad: float = 2) -> bool:
    (x1, y1), (x2, y2) = p1, p2
    x0, y0 = r.x + pad, r.y + pad
    x3, y3 = r.x + r.w - pad, r.y + r.h - pad
    if x0 > x3 or y0 > y3:
        return False
    if abs(y1 - y2) < 0.5:                      # horizontal
        if not (y0 < y1 < y3):
            return False
        lo, hi = sorted((x1, x2))
        return not (hi < x0 or lo > x3)
    if abs(x1 - x2) < 0.5:                      # vertical
        if not (x0 < x1 < x3):
            return False
        lo, hi = sorted((y1, y2))
        return not (hi < y0 or lo > y3)
    # diagonal: sample
    steps = int(max(abs(x2 - x1), abs(y2 - y1)) // 8) + 2
    for i in range(steps + 1):
        t = i / steps
        px, py = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
        if x0 < px < x3 and y0 < py < y3:
            return True
    return False


def validate(page: Page) -> list[str]:
    """Cheap invariants that catch the layout mistakes vision would catch."""
    issues: list[str] = []
    solid = [c for c in page.cells if c.kind == "v" and not c.decor and c.w > 0]
    by_id = {c.id: c for c in page.cells}
    for i, a in enumerate(solid):
        for b in solid[i + 1:]:
            if _rect_overlap(a, b, pad=1):
                issues.append(f"overlap: {a.id}({a.value[:22]!r}) x "
                              f"{b.id}({b.value[:22]!r})")
        if a.x < 0 or a.y < 0:
            issues.append(f"neg-origin: {a.id} at ({a.x},{a.y})")
        if a.x + a.w > page.pw or a.y + a.h > page.ph:
            issues.append(f"outside page: {a.id} ends ({a.x+a.w:.0f},"
                          f"{a.y+a.h:.0f}) > ({page.pw},{page.ph})")
    for c in page.cells:
        if c.kind != "e" or c.decor:
            continue
        pts: list[tuple[float, float]] = []
        if c.source and c.source in by_id:
            s = by_id[c.source]
            pts.append((s.x + s.w / 2, s.y + s.h / 2))
        elif c.src_pt:
            pts.append(c.src_pt)
        pts.extend(c.points)
        if c.target and c.target in by_id:
            t = by_id[c.target]
            pts.append((t.x + t.w / 2, t.y + t.h / 2))
        elif c.dst_pt:
            pts.append(c.dst_pt)
        if len(pts) < 2:
            issues.append(f"edge {c.id}: fewer than 2 endpoints")
            continue
        ends = {c.source, c.target}
        for p1, p2 in zip(pts, pts[1:]):
            for r in solid:
                if r.id in ends:
                    continue
                if _seg_hits_rect(p1, p2, r):
                    issues.append(f"edge {c.id} crosses {r.id}"
                                  f"({r.value[:20]!r})")
    return issues


class Doc:
    def __init__(self):
        self.pages: list[Page] = []

    def page(self, name: str, pw: int = 2200, ph: int = 1500) -> Page:
        p = Page(self, name, pw, ph)
        self.pages.append(p)
        return p

    def xml(self) -> str:
        for p in self.pages:
            for b in p.pending:
                b.emit()
            p.pending.clear()
            self._fit(p)
        pages = "".join(p.xml(f"pg{i}") for i, p in enumerate(self.pages))
        return ('<?xml version="1.0" encoding="UTF-8"?>\n'
                f'<mxfile host="drawio" pages="{len(self.pages)}">{pages}</mxfile>')

    @staticmethod
    def _fit(p: Page) -> None:
        """Fit the page box to its content (grow or shrink, keep a margin)."""
        mx = my = 0.0
        for c in p.cells:
            if c.kind == "v":
                mx = max(mx, c.x + c.w)
                my = max(my, c.y + c.h)
        p.pw = max(600, int(mx) + 40)
        p.ph = max(400, int(my) + 40)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(self.xml())


# ---------------------------------------------------------------- builders ---
LINE_H = 15          # px per label line at fontSize 11
ANCHOR_H = 13        # px per anchor line at fontSize 9
PAD = 12


def _char_w(ch: str, f: float) -> float:
    o = ord(ch)
    if o < 0x2E80:
        return f * 0.62
    return f * 1.02


_FONT_CACHE: dict[int, object] = {}


def _font(size: int):
    """Same font file the renderer uses, so both agree on line counts."""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    f = None
    try:                                     # Windows
        from PIL import ImageFont

        f = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", size)
    except Exception:                        # headless / no CJK font
        try:
            from PIL import ImageFont

            f = ImageFont.load_default()
        except Exception:
            f = None
    _FONT_CACHE[size] = f
    return f


def est_lines(text: str, font_size: float, width: float) -> int:
    """How many wrapped lines will this label need inside `width` px?

    Measured with a real font when possible (identical to render_check.py), so
    the generator and the checker cannot disagree about overflow.
    """
    import re
    plain = re.sub(r"<[^>]+>", "", text)
    plain = (plain.replace("&lt;", "<").replace("&gt;", ">")
             .replace("&amp;", "&").replace("&quot;", '"')
             .replace("&#10;", " "))
    if not plain:
        return 1
    size = max(6, int(round(font_size)))
    f = _font(size)
    if f is not None:
        try:
            from PIL import ImageDraw, Image

            meas = ImageDraw.Draw(Image.new("RGB", (8, 8)))
            lines, cur = 1, ""
            for ch in plain:
                if meas.textlength(cur + ch, font=f) > width and cur:
                    lines += 1
                    cur = ch
                else:
                    cur += ch
            return lines
        except Exception:
            pass
    lines, cur = 1, 0.0
    for ch in plain:
        cw = _char_w(ch, font_size)
        if cur + cw > width and cur > 0:
            lines += 1
            cur = cw
        else:
            cur += cw
    return lines


LINE_H = 14.6        # 11px 行高（与 render_check 的 1.32 倍一致）
ANCHOR_H = 12.0      # 9px 行高
PAD = 6


def measure_h(title: str, anchor: str = "", extra: str = "", w: int = 300,
              min_h: int = 34) -> int:
    """Box height for width w: 11px title + optional 9px extra/anchor lines."""
    inner = max(40, w - 30)
    h = PAD + 6 + est_lines(title, 11, inner) * LINE_H
    if extra:
        h += est_lines(extra, 9, inner) * ANCHOR_H
    if anchor:
        h += est_lines(anchor, 9, inner) * ANCHOR_H
    return int(round(max(min_h, h)))



def measure(title_lines: int, anchor_lines: int = 0, extra: int = 0,
            min_h: int = 34) -> int:
    return max(min_h, PAD + title_lines * LINE_H + anchor_lines * ANCHOR_H
               - 4 + extra)


def label(title: str, anchor: str = "", extra: str = "") -> tuple[str, int, int]:
    """Compose an HTML label with a small grey code-anchor block.
    Returns (html, title_line_count, anchor_line_count)."""
    parts = [title]
    nl = title.count("<br>") + 1
    na = 0
    if extra:
        parts.append(f'<font style="font-size:9px" color="#8a6d3b">{extra}</font>')
        na += extra.count("<br>") + 1
    if anchor:
        na += anchor.count("<br>") + 1
        parts.append(f'<font style="font-size:9px" color="#666666">{anchor}</font>')
    return "<br>".join(parts), nl, na


def box_for(page: Page, x, y, w, title, anchor="", extra="", st="box",
            role=None, bold_top=False, min_h=34, h=None) -> tuple[str, int]:
    lab, _, _ = label(title, anchor, extra)
    hh = h if h is not None else measure_h(title, anchor, extra, w, min_h)
    cid = page.box(x, y, w, hh, lab, st, role, bold_top=bold_top)
    return cid, hh


def table(page: Page, x, y, cols, rows, row_h=30, header=True,
          header_role="ctrl", font=11, col_roles=None, role_matrix=None):
    """Simple grid: cols = [(title, width), ...]; rows = [[cell, ...], ...].

    A cell may be a plain string or (text, role) to colour that one cell.
    Returns the bottom y."""
    cy = y
    for ci, (title, w) in enumerate(cols):
        cx = x + sum(c[1] for c in cols[:ci])
        page.box(cx, cy, w, row_h, f"<b>{title}</b>", "box_center",
                 header_role if header else None, decor=False)
    cy += row_h
    for ri, row in enumerate(rows):
        for ci, cell in enumerate(row):
            cx = x + sum(c[1] for c in cols[:ci])
            role = None
            if isinstance(cell, tuple):
                cell, role = cell
            elif role_matrix:
                role = role_matrix(ri, ci)
            page.box(cx, cy, cols[ci][1], row_h,
                     f'<font style="font-size:{font}px">{cell}</font>',
                     "box", role if role is not None else "grey")
        cy += row_h
    return cy


def legend(page: Page, x, y, items, w=150, h=24, gap=8):
    """items: [(text, role), ...] laid out horizontally."""
    cx = x
    for text, role in items:
        page.box(cx, y, w, h, text, "box_center", role)
        cx += w + gap
    return cx


def legend_bar(page: Page, y: int, extra_fill=None, extra_line=None,
               x0: int = 36):
    """The one legend used by every page: same shapes, order and wording.

    Row 1 = 填色（角色），Row 2 = 线条（平面），可选额外项说明本页专有语义。
    图例**永远显示规范色**：临时清掉本页的 fill_map，否则像 P02/P05/P09 这种
    "把进程方框统一改用灰阶"的页面会把图例本身也改白，造成各页图例不一致。
    """
    saved = dict(page.fill_map)
    page.fill_map = {}
    try:
        page.text(x0, y, 104, 20, "<b>图例</b>", "text_b")
        cx = x0 + 112
        for text, role in LEGEND_ITEMS:
            page.box(cx, y - 3, 120, 26, text, "box_center", role)
            cx += 128
        if extra_fill:
            page.box(cx, y - 3, 240, 26, extra_fill[0], "box_center",
                     extra_fill[1])
        y2 = y + 34
        cx = x0 + 112
        for text, role in LEGEND_LINES:
            page.edge(src_pt=(cx, y2 + 12), dst_pt=(cx + 56, y2 + 12),
                      st="arrow_thick", role=role)
            page.text(cx + 62, y2, 190, 24, text, "subtitle")
            cx += 250
        if extra_line:
            page.edge(src_pt=(cx, y2 + 12), dst_pt=(cx + 56, y2 + 12),
                      st="arrow_dash_open", role=extra_line[1])
            page.text(cx + 62, y2, 200, 24, extra_line[0], "subtitle")
    finally:
        page.fill_map = saved
    return y2 + 30
