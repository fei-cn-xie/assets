"""Swimlane sequence layout + vertical call-chain layout."""
from __future__ import annotations

from dg import C, Page, box_for, label, measure_h, STYLES

HDR_H = 46            # lane header height
TOP = 0               # lane container top
LANE_W = 300
GAP = 18
LEFT = 36
MSG_H = 46
SELF_H = 62
PHASE_H = 28
STEP_GAP = 10


class Seq:
    """Time flows down; each participant is a lane (column).

    Rows: step (one or more boxes in the same time slot), msg (an arrow
    between two lane centres), note (a note box spanning lanes), phase (a
    dashed band that groups the rows below it).
    """

    def __init__(self, page: Page, lanes, x0=LEFT, y0=110, lane_w=LANE_W,
                 title="", subtitle="", show_lifeline=True):
        self.page = page
        self.x0, self.y0 = x0, y0
        self.lane_w = lane_w
        self.show_lifeline = show_lifeline
        self.rows: list[dict] = []
        self.lanes = lanes                     # [(key, label, role)]
        self.key2i = {k: i for i, (k, _, _) in enumerate(lanes)}
        self.left_pad = 0
        self.title = title
        self.subtitle = subtitle
        self.widths = [lane_w] * len(lanes)
        self._phase_stack: list[dict] = []
        page.pending.append(self)

    # ---- geometry ----------------------------------------------------------
    def lane_x(self, i: int) -> float:
        return self.x0 + i * (self.lane_w + GAP)

    def lane_cx(self, i: int) -> float:
        return self.lane_x(i) + self.lane_w / 2

    def idx(self, key) -> int:
        return key if isinstance(key, int) else self.key2i[key]

    # ---- rows --------------------------------------------------------------
    def phase(self, text: str, color: str = "#999999"):
        self.rows.append({"kind": "phase", "text": text, "color": color})

    def gap(self, h: int = 16):
        self.rows.append({"kind": "gap", "h": h})

    def step(self, lane, title: str, anchor: str = "", extra: str = "",
             role=None, st="box", bold_top=True, h=None):
        self.rows.append({"kind": "step",
                          "items": [(lane, title, anchor, extra, role, st,
                                     bold_top, h)]})

    def steps(self, items):
        """items: [(lane, title, anchor, extra, role, st, bold_top, h), ...]"""
        norm = []
        for it in items:
            it = tuple(it) + (None,) * (8 - len(it))
            norm.append(it)
        self.rows.append({"kind": "step", "items": norm})

    def msg(self, a, b, text: str = "", mode="sync", role="ctrl",
            extra: str = "", dashed=False):
        self.rows.append({"kind": "msg", "a": self.idx(a), "b": self.idx(b),
                          "text": text, "mode": mode, "role": role,
                          "extra": extra, "dashed": dashed})

    def selfmsg(self, lane, text: str, role="ctrl", mode="sync"):
        self.rows.append({"kind": "self", "a": self.idx(lane), "text": text,
                          "role": role, "mode": mode})

    def note(self, span, text: str, role=None, st="note", h=None):
        if isinstance(span, (str, int)):
            span = (span, span)
        a, b = self.idx(span[0]), self.idx(span[1])
        self.rows.append({"kind": "note", "a": min(a, b), "b": max(a, b),
                          "text": text, "role": role, "st": st, "h": h})

    def code(self, span, text: str):
        """A slim monospace strip (call-stack / config evidence)."""
        self.note(span, text, role="code", st="code", h=14 + 13 * (text.count("<br>") + 1))

    # ---- layout + emit -----------------------------------------------------
    def emit(self) -> int:
        page = self.page
        n = len(self.lanes)
        # header + title
        if self.title:
            page.text(self.x0, self.y0 - 92, 1500, 26, self.title, "title")
        if self.subtitle:
            page.text(self.x0, self.y0 - 64, 1800, 20, self.subtitle, "subtitle")

        # lane background + header
        content_bottom = self._layout_rows(dry=True) + 40
        for i, (k, lab, role) in enumerate(self.lanes):
            x = self.lane_x(i)
            page.box(x, self.y0, self.lane_w, content_bottom - self.y0, "",
                     "lane", role)
            lab_html = f"<b>{lab}</b>"
            hdr_h = max(26, measure_h(lab_html, "", "", self.lane_w))
            box_for(page, x, self.y0 + 4, self.lane_w, lab_html, st="head",
                    role=role, h=hdr_h)
            if self.show_lifeline:
                page.edge(src_pt=(self.lane_cx(i), self.y0 + 4 + hdr_h + 6),
                          dst_pt=(self.lane_cx(i), content_bottom - 12),
                          st="lifeline", decor=True)
        self._layout_rows(dry=False)
        return content_bottom

    def _row_height(self, r: dict) -> tuple[float, list[float]]:
        """Return (row height, per-item heights), wrap-aware."""
        inner = self.lane_w - 16
        kind = r["kind"]
        if kind == "phase":
            return PHASE_H, []
        if kind == "gap":
            return r["h"], []
        if kind == "step":
            hs = []
            for it in r["items"]:
                lane, title, anchor, extra, role, st, bold_top, h = it
                hs.append(h if h is not None else
                          measure_h(title, anchor or "", extra or "", inner))
            return max(hs) + STEP_GAP, hs
        if kind == "msg":
            return MSG_H, []
        if kind == "self":
            return SELF_H, []
        if kind == "note":
            w = (self.lane_x(r["b"]) + self.lane_w - 8) - (self.lane_x(r["a"]) + 8)
            h = r["h"] if r["h"] is not None else measure_h(r["text"], "", "", w,
                                                            min_h=26)
            return h + STEP_GAP, [h]
        return 0, []

    def _layout_rows(self, dry: bool) -> float:
        page = self.page
        y = self.y0 + HDR_H + 18
        for r in self.rows:
            h, hs = self._row_height(r)
            kind = r["kind"]
            if kind == "phase":
                if not dry:
                    x_end = self.lane_x(len(self.lanes) - 1) + self.lane_w
                    page.box(self.x0, y + 6, x_end - self.x0, PHASE_H - 10,
                             f'<font color="{r["color"]}">{r["text"]}</font>',
                             "phase")
                    page.edge(src_pt=(self.x0, y), dst_pt=(x_end, y), st="bracket",
                              decor=True)
                y += h
            elif kind == "gap":
                y += h
            elif kind == "step":
                for i, (lane, title, anchor, extra, role, st, bold_top, hh) in \
                        enumerate(r["items"]):
                    li = self.idx(lane)
                    lab, nl, na = label(title, anchor, extra)
                    hgt = hs[i]
                    if not dry:
                        box_for(page, self.lane_x(li) + 8, y, self.lane_w - 16,
                                title, anchor, extra, st=st, role=role,
                                bold_top=bold_top, h=hgt)
                y += h
            elif kind == "msg":
                a, b = r["a"], r["b"]
                ym = y + MSG_H / 2 - 6
                style = {"sync": "arrow", "async": "arrow_dash_open",
                         "return": "arrow_dash", "data": "arrow_thick"}[r["mode"]]
                if not dry:
                    xa, xb = self.lane_cx(a), self.lane_cx(b)
                    if abs(xa - xb) < 1:
                        xb = xa + 60
                    direction = 1 if xb > xa else -1
                    txt = r["text"]
                    if r.get("extra"):
                        txt += f'<br><font style="font-size:9px">{r["extra"]}</font>'
                    page.edge(src_pt=(xa, ym), dst_pt=(xb, ym), label=txt,
                              st=style, role=r["role"], label_pos=0)
                y += h
            elif kind == "self":
                if not dry:
                    x = self.lane_cx(r["a"])
                    ym = y + 16
                    page.edge(src_pt=(x, ym), dst_pt=(x, ym + 22),
                              points=[(x + 92, ym), (x + 92, ym + 22)],
                              label=r["text"].replace("<br>", "<br>"),
                              st="arrow", role=r["role"], label_pos=0)
                y += h
            elif kind == "note":
                if not dry:
                    a, b = r["a"], r["b"]
                    x1 = self.lane_x(a) + 8
                    x2 = self.lane_x(b) + self.lane_w - 8
                    page.box(x1, y, x2 - x1, hs[0],
                             label(r["text"])[0], r["st"],
                             r["role"])
                y += h
        return y


# ------------------------------------------------------------------ chains ---
def chain(page: Page, x, y, w, steps, role=None, gap=30, arrow_label="",
          st="arrow", labels=None):
    """Vertical call chain. steps: [(title, anchor, extra) | (title, anchor) | str]

    Returns (ids, next_y)."""
    ids: list[str] = []
    cy = y
    for i, s in enumerate(steps):
        if isinstance(s, str):
            title, anchor, extra = s, "", ""
        else:
            title, anchor, extra = (tuple(s) + ("", ""))[:3]
        cid, hh = box_for(page, x, cy, w, title, anchor, extra, role=role)
        ids.append(cid)
        cy += hh
        if i != len(steps) - 1:
            cy += gap
    for i in range(len(ids) - 1):
        lab = ""
        if labels and i < len(labels) and labels[i]:
            lab = labels[i]
        elif i == 0:
            lab = arrow_label
        page.edge(ids[i], ids[i + 1], label=lab, st=st, role=role)
    return ids, cy


def aligned(page: Page, rows, x1, w1, gap=8, sides=None, x3=None, w3=0,
            role=None, side_role="grey", third_role="code", y=110,
            link=True):
    """Two-column aligned rows: left = call-chain step, right = annotation.

    rows: [(title, anchor, extra), ...]      (main column labels)
    sides: [text or None, ...]               (aligned annotation per row)
    Returns (ids, side_ids, next_y).
    """
    ids, side_ids = [], []
    cy = y
    heights = []
    for r in rows:
        title, anchor, extra = (tuple(r) + ("", "", ""))[:3]
        heights.append(measure_h(title or "", anchor or "", extra or "", w1))
    for i, (r, h) in enumerate(zip(rows, heights)):
        title, anchor, extra = (tuple(r) + ("", "", ""))[:3]
        cid, hh = box_for(page, x1, cy, w1, title or "", anchor or "", extra or "",
                          role=role, h=h)
        ids.append(cid)
        if sides and i < len(sides) and sides[i]:
            sh = max(hh, measure_h(sides[i], "", "", 460, min_h=26))
            sid = page.box(x1 + w1 + 40, cy, 460, sh, label(sides[i])[0], "note",
                           side_role)
            side_ids.append(sid)
        cy += hh + gap
    if link:
        for i in range(len(ids) - 1):
            page.edge(ids[i], ids[i + 1], st="arrow", role=role)
    return ids, side_ids, cy
