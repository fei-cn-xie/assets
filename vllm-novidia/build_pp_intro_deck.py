# -*- coding: utf-8 -*-
"""Build a 16-slide, beginner-first deck about pipeline parallelism (PP) in vLLM.

Scope: vllm main branch, NVIDIA GPU (CUDA/NCCL) path only.
All diagrams are native PowerPoint shapes (editable, no raster images).
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

# ----------------------------------------------------------------- palette
INK = RGBColor(0x0F, 0x1B, 0x2D)
SLATE = RGBColor(0x3B, 0x4A, 0x63)
MUTED = RGBColor(0x6B, 0x7A, 0x90)
HAIR = RGBColor(0xDC, 0xE3, 0xEC)
PANEL = RGBColor(0xF6, 0xF8, 0xFB)
PANEL2 = RGBColor(0xEE, 0xF3, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

BLUE = RGBColor(0x25, 0x63, 0xEB)
TEAL = RGBColor(0x0E, 0x93, 0x84)
VIOLET = RGBColor(0x7C, 0x3A, 0xED)
ORANGE = RGBColor(0xEA, 0x58, 0x0C)
AMBER = RGBColor(0xD9, 0x77, 0x06)
RED = RGBColor(0xDC, 0x26, 0x26)
GREEN = RGBColor(0x15, 0x80, 0x3D)

SOFT = {
    "blue": RGBColor(0xDC, 0xE7, 0xFB),
    "teal": RGBColor(0xD6, 0xF0, 0xED),
    "violet": RGBColor(0xEC, 0xE4, 0xFB),
    "orange": RGBColor(0xFD, 0xE7, 0xDA),
    "amber": RGBColor(0xFE, 0xF3, 0xC7),
    "red": RGBColor(0xFD, 0xE2, 0xE2),
    "green": RGBColor(0xDC, 0xFC, 0xE7),
    "grey": RGBColor(0xEE, 0xF2, 0xF7),
}
STAGE = [BLUE, TEAL, VIOLET, ORANGE]
STAGE_SOFT = [SOFT["blue"], SOFT["teal"], SOFT["violet"], SOFT["orange"]]
STAGE_NAME = ["S0", "S1", "S2", "S3"]

FONT = "微软雅黑"
MONO = "Consolas"

SW, SH = 13.333, 7.5
ML, MR = 0.62, 0.62
CW = SW - ML - MR

prs = Presentation()
prs.slide_width = Inches(SW)
prs.slide_height = Inches(SH)
BLANK = prs.slide_layouts[6]

SRC = "vllm@main · 70c0016 · 仅 NVIDIA GPU 路径"


# ----------------------------------------------------------------- helpers
def _font(run, name):
    run.font.name = name
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = etree.SubElement(rPr, qn(tag))
        el.set("typeface", name)


def _no_autofit(tf):
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE


def _bullet(p, char="▪", color=MUTED, marL=0.22, indent=-0.22, pct=90):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(Inches(marL))))
    pPr.set("indent", str(int(Inches(indent))))
    clr = etree.SubElement(pPr, qn("a:buClr"))
    srgb = etree.SubElement(clr, qn("a:srgbClr"))
    srgb.set("val", str(color))
    sz = etree.SubElement(pPr, qn("a:buSzPct"))
    sz.set("val", str(int(pct * 1000)))
    fnt = etree.SubElement(pPr, qn("a:buFont"))
    fnt.set("typeface", "Arial")
    ch = etree.SubElement(pPr, qn("a:buChar"))
    ch.set("char", char)


def rich(slide, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP, margins=(0.02, 0.02, 0.02, 0.02)):
    """paras: list of dicts {runs:[(text, {size,color,bold,font})], align, after, before,
    line, bullet:(char,color) | None, marL}"""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    _no_autofit(tf)
    tf.vertical_anchor = anchor
    tf.margin_left = Inches(margins[0])
    tf.margin_right = Inches(margins[1])
    tf.margin_top = Inches(margins[2])
    tf.margin_bottom = Inches(margins[3])
    for i, spec in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = spec.get("align", PP_ALIGN.LEFT)
        p.line_spacing = spec.get("line", 1.08)
        p.space_before = Pt(spec.get("before", 0))
        p.space_after = Pt(spec.get("after", 4))
        if spec.get("bullet"):
            ch, col = spec["bullet"]
            _bullet(p, ch, col, marL=spec.get("marL", 0.22), indent=-spec.get("marL", 0.22))
        for text, opt in spec["runs"]:
            r = p.add_run()
            r.text = text
            r.font.size = Pt(opt.get("size", 14))
            r.font.bold = opt.get("bold", False)
            r.font.italic = opt.get("italic", False)
            r.font.color.rgb = opt.get("color", INK)
            _font(r, opt.get("font", FONT))
    return box


def txt(slide, x, y, w, h, s, size=14, color=INK, bold=False, align=PP_ALIGN.LEFT,
        anchor=MSO_ANCHOR.TOP, font=FONT, line=1.08, after=0, margins=(0.02, 0.02, 0.02, 0.02)):
    parts = s.split("\n")
    paras = [{"runs": [(p, {"size": size, "color": color, "bold": bold, "font": font})],
              "align": align, "line": line, "after": after if i == len(parts) - 1 else 0}
             for i, p in enumerate(parts)]
    return rich(slide, x, y, w, h, paras, anchor=anchor, margins=margins)


def shape(slide, x, y, w, h, fill=None, line=None, lw=1.0, radius=0.10, kind=MSO_SHAPE.ROUNDED_RECTANGLE):
    sp = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    # drop the theme style reference (removes inherited shadow/outline effects)
    style = sp._element.find(qn("p:style"))
    if style is not None:
        sp._element.remove(style)
    if kind == MSO_SHAPE.ROUNDED_RECTANGLE and radius is not None:
        try:
            sp.adjustments[0] = radius
        except Exception:
            pass
    if fill is None:
        sp.fill.background()
    else:
        sp.fill.solid()
        sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line
        sp.line.width = Pt(lw)
    sp.shadow.inherit = False
    sp.text_frame.word_wrap = True
    return sp


def label(sp, s, size=11, color=INK, bold=False, align=PP_ALIGN.CENTER, font=FONT,
          margins=(0.03, 0.03, 0.01, 0.01)):
    tf = sp.text_frame
    _no_autofit(tf)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(margins[0])
    tf.margin_right = Inches(margins[1])
    tf.margin_top = Inches(margins[2])
    tf.margin_bottom = Inches(margins[3])
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = 1.0
    r = p.add_run()
    r.text = s
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    _font(r, font)
    return sp


def arrow(slide, x1, y1, x2, y2, color=MUTED, lw=1.4, dash=None, head=True, tail=False,
          head_size="med"):
    conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                                      Inches(x2), Inches(y2))
    conn.line.color.rgb = color
    conn.line.width = Pt(lw)
    if dash:
        conn.line.dash_style = dash
    ln = conn.line._get_or_add_ln()
    if head:
        e = etree.SubElement(ln, qn("a:tailEnd"))
        e.set("type", "triangle")
        e.set("w", head_size)
        e.set("len", head_size)
    if tail:
        e = etree.SubElement(ln, qn("a:headEnd"))
        e.set("type", "triangle")
        e.set("w", head_size)
        e.set("len", head_size)
    return conn


def chip(slide, x, y, w, h, s, fill=PANEL2, color=SLATE, size=10, bold=True, line=None):
    sp = shape(slide, x, y, w, h, fill=fill, line=line, lw=0.9, radius=0.3)
    label(sp, s, size=size, color=color, bold=bold)
    return sp


# ----------------------------------------------------------------- chrome
# 每页右上角的“承接”标注:说明这一页从上一页的哪个结论接过来
BRIDGE = {
    2: "承接 P1 · 先给全景与路线",
    3: "承接 P2 · 01 缘起:为什么需要并行",
    4: "承接 P3 · 装不下 → 能不能按层拆?",
    5: "承接 P4 · 已确定「按层切」,看它站哪一格",
    6: "承接 P5 · 切层之后,计算变成串行",
    7: "承接 P6 · 气泡从哪来 → 怎么填",
    8: "承接 P7 · 机制清楚了,看 vLLM 怎么落地",
    9: "承接 P8 · 二维卡位 → 三维怎么排",
    10: "承接 P9 · 卡位定了,层具体怎么切",
    11: "承接 P10 · 切开之后,段与段怎么接上",
    12: "承接 P11 · 传的是 hidden,采样怎么回传",
    13: "承接 P12 · 三条线索合成一次 forward",
    14: "承接 P13 · 跑通了,算算显存这笔账",
    15: "承接 P14 · 账算清了,怎么开起来",
    16: "承接 P15 · 能开起来,再说哪里会踩坑",
    17: "承接 P16 · 收束成三件事与源码索引",
    18: "承接 P17 · 想深入,看这些文档与论文",
}


def new_slide(section_color, kicker, title, lead=None):
    s = prs.slides.add_slide(BLANK)
    page = len(prs.slides)
    bar = shape(s, 0, 0, SW, 0.075, fill=section_color, kind=MSO_SHAPE.RECTANGLE)
    bar.line.fill.background()
    sq = shape(s, ML, 0.375, 0.115, 0.115, fill=section_color, kind=MSO_SHAPE.RECTANGLE)
    sq.line.fill.background()
    txt(s, ML + 0.22, 0.315, 8.6, 0.26, kicker, size=10.5, color=section_color, bold=True)
    bridge = BRIDGE.get(page)
    if bridge:
        txt(s, SW - MR - 4.30, 0.315, 4.30, 0.24, "↳ " + bridge, size=8.6,
            color=RGBColor(0x82, 0x90, 0xA4), align=PP_ALIGN.RIGHT)
    txt(s, ML, 0.60, CW, 0.62, title, size=26, color=INK, bold=True)
    if lead:
        txt(s, ML, 1.24, CW, 0.34, lead, size=13, color=MUTED)
    return s


def footer(s, page, ref=None):
    ln = shape(s, ML, 7.03, CW, 0.012, fill=HAIR, kind=MSO_SHAPE.RECTANGLE)
    ln.line.fill.background()
    txt(s, ML, 7.10, 9.6, 0.24, ref or SRC, size=8.5, color=MUTED)
    txt(s, SW - MR - 0.8, 7.10, 0.8, 0.24, f"{page:02d}", size=8.5, color=MUTED,
        align=PP_ALIGN.RIGHT, bold=True)


def notes(s, text):
    s.notes_slide.notes_text_frame.text = text


# ----------------------------------------------------------------- widgets
def code_card(s, x, y, w, h, title, lines, size=9.6, hl=None, title_color=AMBER):
    box = shape(s, x, y, w, h, fill=RGBColor(0x11, 0x1B, 0x2B), radius=0.05)
    box.line.color.rgb = RGBColor(0x11, 0x1B, 0x2B)
    if title:
        txt(s, x + 0.18, y + 0.11, w - 0.36, 0.22, title, size=8.8, color=title_color,
            bold=True, font=MONO)
        ty = y + 0.40
    else:
        ty = y + 0.14
    paras = []
    for i, line in enumerate(lines):
        col = RGBColor(0xE4, 0xEA, 0xF2)
        if hl and i in hl:
            col = hl[i]
        paras.append({"runs": [(line if line else " ", {"size": size, "color": col, "font": MONO})],
                      "line": 1.12, "after": 0})
    rich(s, x + 0.16, ty, w - 0.30, y + h - ty - 0.10, paras, margins=(0.02, 0.02, 0.02, 0.0))


def timeline(s, x, y, w, rows, ncols, cells, row_h=0.40, gap=0.05, label_w=1.22,
             time_label="时间 →", caption=None, cell_w=None):
    """cells: {(row, col): (kind, text)} kind in compute/idle/comm/wait."""
    avail = w - label_w
    cw = cell_w or (avail - (ncols - 1) * gap) / ncols
    for r in range(rows):
        txt(s, x, y + r * (row_h + gap) + (row_h - 0.2) / 2, label_w - 0.08, 0.22,
            f"Stage {r}", size=9.5, color=STAGE[r] if r < 4 else SLATE, bold=True,
            align=PP_ALIGN.RIGHT)
        for c in range(ncols):
            cx = x + label_w + c * (cw + gap)
            kind, text = cells.get((r, c), ("idle", ""))
            if kind == "compute":
                sp = shape(s, cx, y + r * (row_h + gap), cw, row_h, fill=STAGE[r], radius=0.12)
                if text:
                    label(sp, text, size=8.5, color=WHITE, bold=True, margins=(0.0, 0.0, 0.0, 0.0))
            elif kind == "comm":
                sp = shape(s, cx, y + r * (row_h + gap), cw, row_h,
                           fill=SOFT["amber"], line=AMBER, lw=0.9, radius=0.12)
                if text:
                    label(sp, text, size=8, color=AMBER, bold=True, margins=(0.0, 0.0, 0.0, 0.0))
            elif kind == "wait":
                sp = shape(s, cx, y + r * (row_h + gap), cw, row_h,
                           fill=WHITE, line=HAIR, lw=0.9, radius=0.12)
                if text:
                    label(sp, text, size=8, color=MUTED, margins=(0.0, 0.0, 0.0, 0.0))
            else:
                sp = shape(s, cx, y + r * (row_h + gap), cw, row_h, fill=RGBColor(0xEC, 0xF0, 0xF6),
                           radius=0.12)
                if text:
                    label(sp, text, size=8, color=RGBColor(0x9A, 0xA6, 0xB6),
                          margins=(0.0, 0.0, 0.0, 0.0))
    if time_label:
        txt(s, x + label_w, y + rows * (row_h + gap) + 0.04, w - label_w, 0.22, time_label,
            size=8.5, color=MUTED, align=PP_ALIGN.RIGHT)
    if caption:
        txt(s, x, y + rows * (row_h + gap) + 0.04, w - 1.2, 0.22, caption, size=9.5, color=MUTED)


def table(s, x, y, w, data, col_w, row_h=0.36, header_h=0.40, size=10.5, header_size=10.5,
          header_fill=INK, zebra=(WHITE, PANEL), first_col_bold=True, aligns=None,
          valign=MSO_ANCHOR.MIDDLE, header_color=WHITE):
    nrows, ncols = len(data), len(data[0])
    gf = s.shapes.add_table(nrows, ncols, Inches(x), Inches(y), Inches(w),
                            Inches(header_h + row_h * (nrows - 1)))
    tbl = gf.table
    tbl.first_row = True
    tbl.horz_banding = False
    tbl.vert_banding = False
    # "No Style, No Grid"
    tblPr = tbl._tbl.tblPr
    for el in tblPr.findall(qn("a:tableStyleId")):
        tblPr.remove(el)
    sid = etree.SubElement(tblPr, qn("a:tableStyleId"))
    sid.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"
    for i, cw in enumerate(col_w):
        tbl.columns[i].width = Inches(cw)
    tbl.rows[0].height = Inches(header_h)
    for r in range(1, nrows):
        tbl.rows[r].height = Inches(row_h)
    for r, row in enumerate(data):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.margin_left = Inches(0.10)
            cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = valign
            cell.fill.solid()
            if r == 0:
                cell.fill.fore_color.rgb = header_fill
            else:
                cell.fill.fore_color.rgb = zebra[(r - 1) % 2]
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.line_spacing = 1.0
            if aligns:
                p.alignment = aligns[c]
            runs = val if isinstance(val, list) else [(str(val), {})]
            for text, opt in runs:
                run = p.add_run()
                run.text = text
                run.font.size = Pt(opt.get("size", header_size if r == 0 else size))
                run.font.bold = opt.get("bold", header_size > 0 and r == 0) or (
                    first_col_bold and c == 0 and r > 0)
                run.font.color.rgb = opt.get("color", header_color if r == 0 else INK)
                _font(run, opt.get("font", MONO if opt.get("mono") else FONT))
                if opt.get("link"):
                    run.hyperlink.address = opt["link"]
    return gf


# ----------------------------------------------------------------- slide 1
def s01():
    s = prs.slides.add_slide(BLANK)
    bar = shape(s, 0, 0, 0.10, SH, fill=BLUE, kind=MSO_SHAPE.RECTANGLE)
    bar.line.fill.background()

    txt(s, ML + 0.16, 2.24, 8.4, 0.28, "推理系统 · 分布式并行", size=11, color=BLUE, bold=True)
    txt(s, ML + 0.16, 2.62, 9.6, 1.05, "PP 流水线并行", size=60, color=INK, bold=True)
    txt(s, ML + 0.16, 3.78, 9.6, 0.52, "从「一张卡放不下」到 vLLM 的 GPU 实现", size=21, color=SLATE)
    rule = shape(s, ML + 0.16, 4.48, 1.35, 0.05, fill=ORANGE, kind=MSO_SHAPE.RECTANGLE)
    rule.line.fill.background()

    txt(s, ML + 0.16, 6.40, 8.4, 0.28,
        "vllm main @ 70c0016 · 仅 NVIDIA GPU(CUDA / NCCL)路径",
        size=9, color=MUTED)
    notes(s, "开场:只讲 vLLM main 分支 GPU 路径下的 PP。目标是把『为什么要 PP、PP 怎么跑、"
             "vLLM 怎么实现、怎么开怎么排错』讲清楚,最后带走一张代码地图。")


# ----------------------------------------------------------------- slide 2
def s02():
    s = new_slide(BLUE, "内容结构", "从并行动因到部署边界",
                  "五个板块:缘起 → 直觉 → 机制 → 实现 → 部署;每个结论都能在仓库里找到出处。")
    items = [
        ("01", "缘起", "2018 – 2019", "模型比显存长得快;DP 复制模型、TP 又把通信压在 NVLink 上 —— 跨机部署需要第三种解法。", BLUE, "P3 – P4"),
        ("02", "直觉", "切层与接力", "把 N 层切成 p 段,一段一张卡;段与段之间只传 hidden states。和 DP/TP/EP 的本质差别。", TEAL, "P5"),
        ("03", "机制", "气泡与填充", "串行接力会让卡空转。气泡从哪来、(p−1)/(m+p−1) 怎么算、vLLM 用什么把流水线填满。", VIOLET, "P6 – P7"),
        ("04", "实现", "vLLM GPU 路径", "命令如何变成进程;DP/TP/PP 怎么排卡;层怎么切、张量怎么传、采样怎么回传。", ORANGE, "P8 – P14"),
        ("05", "部署", "选型与边界", "三条部署命令、官方选型建议、7 个硬限制,以及最常见的挂起原因和排查手法。", INK, "P15 – P16"),
    ]
    x = ML
    cw, gap = 2.30, 0.115
    for num, name, sub, desc, col, page in items:
        shape(s, x, 1.72, cw, 2.92, fill=WHITE, line=HAIR, lw=1.0, radius=0.05)
        shape(s, x, 1.72, cw, 0.055, fill=col, kind=MSO_SHAPE.RECTANGLE).line.fill.background()
        txt(s, x + 0.18, 1.86, 0.9, 0.4, num, size=22, color=col, bold=True)
        txt(s, x + 0.18, 2.30, cw - 0.3, 0.3, name, size=14.5, color=INK, bold=True)
        txt(s, x + 0.18, 2.60, cw - 0.3, 0.24, sub, size=9, color=col)
        txt(s, x + 0.18, 2.92, cw - 0.34, 1.34, desc, size=9.8, color=SLATE, line=1.28)
        txt(s, x + 0.18, 4.30, cw - 0.3, 0.22, page, size=8, color=MUTED, bold=True)
        x += cw + gap

    txt(s, ML, 4.90, 4.2, 0.28, "三个核心问题", size=11, color=INK, bold=True)
    qs = [("该不该上 PP", "模型大小、卡数、有没有 NVLink、请求并发是否够"),
          ("vLLM 里怎么跑", "切层、点对点收发(P2P)、采样回传分别落在哪段代码"),
          ("出问题怎么查", "从日志、batch 一致性、NCCL 三层定位挂起")]
    x = ML
    for q, a in qs:
        shape(s, x, 5.24, 3.86, 1.16, fill=PANEL, radius=0.06).line.color.rgb = PANEL
        txt(s, x + 0.16, 5.36, 3.55, 0.24, q, size=11.5, color=BLUE, bold=True)
        txt(s, x + 0.16, 5.66, 3.55, 0.62, a, size=9.5, color=SLATE, line=1.18)
        x += 3.99
    footer(s, 2)
    notes(s, "先给结构:五个板块。前两个板块不需要看代码;第三个板块会打开 vLLM 的 8 个文件;"
             "第四部分是命令和排错。如果时间紧,可直接看 P8–P14 的实现部分。")


# ----------------------------------------------------------------- slide 3
def s03():
    s = new_slide(BLUE, "01 · 缘起  为什么会有 PP",
                  "问题不是「算得慢」,而是「根本装不下」",
                  "2018 年至今,参数量的增长远超单卡显存:一张 80GB 的 H100,连一半权重都放不下。")
    stats = [("2018", "GPT-1", "117M", "单卡轻松放下"),
             ("2019", "GPT-2", "1.5B", "fp16 约 3GB"),
             ("2020", "GPT-3", "175B", "fp16 约 350GB"),
             ("2023 →", "MoE 时代", "千亿 ~ 万亿", "fp16 可达数 TB")]
    x = ML
    for year, name, size_, note in stats:
        shape(s, x, 1.78, 2.92, 1.16, fill=PANEL, radius=0.05).line.color.rgb = PANEL
        txt(s, x + 0.16, 1.86, 1.0, 0.24, year, size=9.5, color=BLUE, bold=True)
        txt(s, x + 0.16, 2.10, 1.5, 0.24, name, size=10, color=SLATE)
        txt(s, x + 1.35, 2.02, 1.45, 0.40, size_, size=17, color=INK, bold=True, align=PP_ALIGN.RIGHT)
        txt(s, x + 1.35, 2.44, 1.45, 0.24, note, size=8.5, color=MUTED, align=PP_ALIGN.RIGHT)
        x += 3.05

    # left: memory math
    txt(s, ML, 3.16, 6.2, 0.28, "算一笔账:80GB 的一张卡能装多少?", size=12, color=INK, bold=True)
    rows = [("权重", "fp16 下每 1B 参数 ≈ 2GB → 80GB 只够 ~40B 权重"),
            ("70B 模型", "fp16 权重 = 140GB,一张 80GB 卡放不下"),
            ("KV Cache", "70B 每 token ≈ 320KB(fp16,全模型)")]
    y = 3.48
    for k, v in rows:
        shape(s, ML, y, 6.05, 0.58, fill=WHITE, line=HAIR, lw=0.9, radius=0.05)
        txt(s, ML + 0.16, y + 0.16, 1.15, 0.26, k, size=10.5, color=BLUE, bold=True)
        txt(s, ML + 1.38, y + 0.15, 4.55, 0.3, v, size=10, color=SLATE)
        y += 0.65

    # right: three ways out
    txt(s, 6.98, 3.16, 5.7, 0.28, "三条出路:只有第三条能横向扩展", size=12, color=INK, bold=True)
    outs = [("① 改模型", "量化 / 稀疏 / 蒸馏:能变小,但精度和效果要重新验证", MUTED),
            ("② 换内存", "CPU offload / 分层换入换出:能装下,但带宽掉一个量级", MUTED),
            ("③ 切到多卡", "并行:TP 切层内、PP 切层间 —— 唯一能随卡数线性扩展的路", GREEN)]
    y = 3.48
    for k, v, col in outs:
        shape(s, 6.98, y, 5.72, 0.58, fill=WHITE if col == MUTED else SOFT["green"],
              line=HAIR if col == MUTED else GREEN, lw=0.9, radius=0.05)
        txt(s, 7.14, y + 0.15, 1.32, 0.26, k, size=10.5, color=col if col != MUTED else SLATE, bold=True)
        txt(s, 8.50, y + 0.10, 4.08, 0.42, v, size=9.3, color=SLATE, line=1.1)
        y += 0.65

    strip = shape(s, ML, 5.72, CW, 0.94, fill=SOFT["amber"], line=AMBER, lw=1.0, radius=0.05)
    txt(s, ML + 0.22, 5.84, 1.6, 0.26, "关键区别", size=10, color=AMBER, bold=True)
    rich(s, ML + 1.30, 5.82, CW - 1.55, 0.74, [
        {"runs": [("DP(数据并行)复制整个模型", {"size": 11, "color": INK, "bold": True}),
                  (" —— 每张卡仍要装下一整份权重。它解决的是「吞吐」,不解决「单个模型装不下」。",
                   {"size": 11, "color": SLATE})], "after": 2, "line": 1.15},
        {"runs": [("要把一个模型摊开到更多卡上,只能沿「层」或「层内张量」切 —— 这就是 TP 与 PP 的起点。",
                   {"size": 10.5, "color": SLATE})], "line": 1.15},
    ])
    footer(s, 3, "换算:fp16 每参数 2 字节;KV = 2(K,V)× 8 heads × 128 dim × 2B × 80 层 ≈ 320KB / token")
    notes(s, "先把问题立住:不是算力不够,是权重和 KV 放不进一张卡。DP 解决吞吐不解决容量,"
             "这一点很多初学者会混淆。引出下一步:模型能不能拆?能,因为它是层叠的。")


# ----------------------------------------------------------------- slide 4
def s04():
    s = new_slide(BLUE, "01 · 缘起  2018–2019 的三个里程碑",
                  "关键洞察:模型是「一摞层」,层与层之间只有一个中间张量",
                  "从 Embedding 到 LM Head 是一串串行的层;相邻两层之间流动的,只有 hidden states(+ residual)。")
    # layer chain
    y = 1.92
    blocks = [("Embedding", "词表 × hidden", SOFT["blue"], BLUE),
              ("Transformer 层 × N", "注意力 + MLP", SOFT["teal"], TEAL),
              ("RMSNorm", "归一化", SOFT["violet"], VIOLET),
              ("LM Head", "hidden × 词表", SOFT["orange"], ORANGE)]
    x = ML
    widths = [1.72, 2.45, 1.45, 1.72]
    for (title, sub, fill, col), w in zip(blocks, widths):
        shape(s, x, y, w, 0.94, fill=fill, line=col, lw=1.0, radius=0.06)
        txt(s, x + 0.08, y + 0.20, w - 0.16, 0.28, title, size=10, color=col, bold=True,
            align=PP_ALIGN.CENTER)
        txt(s, x + 0.08, y + 0.52, w - 0.16, 0.26, sub, size=8.2, color=SLATE, align=PP_ALIGN.CENTER)
        x += w
        if x < ML + 7.6:
            arrow(s, x + 0.04, y + 0.47, x + 0.30, y + 0.47, color=MUTED, lw=1.2)
            x += 0.34
    txt(s, ML, y + 1.04, 8.3, 0.26,
        "层与层之间只流动 hidden_states([tokens, hidden])与 residual —— 不是整个模型,也不是权重",
        size=9.2, color=MUTED)

    # right: milestones
    ms = [("GPipe", "2018 / 2019", "把 mini-batch 切成 micro-batch,让多批在阶段间流水起来;给出气泡率公式。", BLUE),
          ("PipeDream", "2019", "异步流水 + 1F1B 调度,进一步压缩气泡(训练场景,需要权重暂存)。", TEAL),
          ("Megatron-LM", "2019", "把 TP(层内)与 PP(层间)组合成标准配方,至今仍是主流拓扑。", VIOLET)]
    y2 = 1.86
    for name, year, desc, col in ms:
        shape(s, 9.30, y2, 3.40, 1.06, fill=WHITE, line=HAIR, lw=0.9, radius=0.05)
        txt(s, 9.44, y2 + 0.10, 2.0, 0.26, name, size=11.5, color=col, bold=True)
        txt(s, 9.44, y2 + 0.36, 2.8, 0.22, year, size=8.5, color=MUTED)
        txt(s, 9.44, y2 + 0.58, 3.10, 0.44, desc, size=8.5, color=SLATE, line=1.12)
        y2 += 1.14

    # insight box
    shape(s, ML, 3.32, 8.36, 1.52, fill=INK, radius=0.05)
    txt(s, ML + 0.22, 3.42, 2.0, 0.26, "于是可以这样切", size=10, color=AMBER, bold=True)
    rich(s, ML + 0.22, 3.68, 7.95, 1.08, [
        {"runs": [("GPU 0 放 L0 – L19,GPU 1 放 L20 – L39,……", {"size": 12.5, "color": WHITE, "bold": True}),
                  ("  每张卡只装一段层,显存压力被摊开。", {"size": 11, "color": RGBColor(0xC7, 0xD2, 0xE0)})],
         "after": 5, "line": 1.2},
        {"runs": [("代价:计算变成串行 —— 同一时刻只有一段在算,其余在等。这个「等」就是下一节的气泡。",
                   {"size": 10.2, "color": RGBColor(0xC7, 0xD2, 0xE0)})], "line": 1.2},
    ])
    txt(s, ML, 5.06, CW, 0.3,
        "PP 的本质不是「算得更快」,而是把模型摊到更多卡上,并尽量不浪费算力。",
        size=12, color=INK, bold=True)
    txt(s, ML, 5.42, CW, 0.9,
        "对比一下:TP 是「把一个矩阵切给多张卡同时算」,PP 是「把不同的层放到不同卡上按顺序算」。"
        "前者省单层显存但有大量集合通信,后者省整段显存且只需要点对点通信 —— 这是 PP 能跨机部署的根本原因。",
        size=10, color=SLATE, line=1.3)
    footer(s, 4, "参考:GPipe(NeurIPS 2019)· PipeDream(SOSP 2019)· Megatron-LM(2019)")
    notes(s, "这一页是全篇的地基:模型是层链,层间只传 hidden states。有了这个结构,才谈得上按层切。"
             "顺便埋下伏笔:切完就变成串行,串行就有气泡。")


# ----------------------------------------------------------------- slide 5
def s05():
    s = new_slide(TEAL, "02 · 直觉  四种并行,PP 站在哪一格",
                  "PP 切的是「层与层之间」,不是「层里面」",
                  "同一条层链,可以横着切(PP)、竖着切(TP);切法不同,通信量和适用场景也完全不同。")
    data = [
        ["维度", "切什么", "通信量", "带宽要求 / 场景"],
        ["DP 数据并行", "复制整个模型,分发请求", "几乎不通信", "低;提吞吐,不省显存"],
        ["TP 张量并行", "切一层内的矩阵", "每层 2 次 all-reduce", "极高;要 NVLink,通常限节点内"],
        ["PP 流水线并行", "切层与层之间", "每个边界 1 次点对点", "中低;可跨机,PCIe/IB 都行"],
        ["EP 专家并行", "切 MoE 的专家", "每层 all-to-all", "高;MoE 专用,常与 DP 组合"],
    ]
    table(s, ML, 1.80, 8.10, data, [1.52, 2.06, 1.98, 2.54], row_h=0.52, header_h=0.42,
          size=9.8, header_size=10)
    # highlight the PP row
    shape(s, ML + 0.02, 1.80 + 0.42 + 2 * 0.52 + 0.01, 8.06, 0.50,
          fill=None, line=TEAL, lw=1.6, radius=0.02, kind=MSO_SHAPE.RECTANGLE)

    # right: TP (vertical split) vs PP (horizontal split)
    base_x = 9.02
    txt(s, base_x, 1.78, 3.6, 0.24, "TP:竖切,每层都要同步一次", size=9.5, color=BLUE, bold=True)
    for r in range(3):
        for c in range(3):
            shape(s, base_x + c * 0.37, 2.08 + r * 0.29, 0.35, 0.27,
                  fill=SOFT["blue"], line=BLUE, lw=0.7, radius=0.12)
        txt(s, base_x + 1.16, 2.12 + r * 0.29, 0.34, 0.2, "⟷", size=9, color=BLUE,
            align=PP_ALIGN.CENTER)
    rich(s, base_x + 1.58, 2.06, 2.05, 0.92, [
        {"runs": [("一层拆成 3 份", {"size": 7.8, "color": SLATE, "bold": True})], "after": 1, "line": 1.05},
        {"runs": [("每层算完都要 all-reduce 拼回完整结果", {"size": 7.6, "color": MUTED})], "line": 1.1},
    ])

    txt(s, base_x, 3.06, 3.6, 0.24, "PP:横切,只在边界传一次", size=9.5, color=TEAL, bold=True)
    for r in range(3):
        bar = shape(s, base_x, 3.36 + r * 0.40, 1.11, 0.25, fill=STAGE_SOFT[r], line=STAGE[r],
                    lw=0.9, radius=0.2)
        label(bar, f"stage {r}", size=7, color=STAGE[r], bold=True)
    for r in range(2):
        arrow(s, base_x + 0.56, 3.62 + r * 0.40, base_x + 0.56, 3.74 + r * 0.40,
              color=TEAL, lw=1.0, head_size="sm")
    rich(s, base_x + 1.58, 3.34, 2.05, 0.96, [
        {"runs": [("每行 = 一个 stage", {"size": 7.8, "color": SLATE, "bold": True})], "after": 1, "line": 1.05},
        {"runs": [("行与行之间只传一次 hidden states", {"size": 7.6, "color": MUTED})], "line": 1.1},
    ])

    # bottom: communication volume
    shape(s, ML, 4.56, CW, 1.00, fill=SOFT["amber"], line=AMBER, lw=1.0, radius=0.05)
    txt(s, ML + 0.22, 4.64, 3.0, 0.24, "量级对比(8K token 批,70B 模型)", size=9.5, color=AMBER, bold=True)
    rich(s, ML + 0.22, 4.88, CW - 0.5, 0.62, [
        {"runs": [("TP=8:每层 2 次 all-reduce,80 层合计 ≈ 21GB;PP=2:整个批只过 1 个边界 ≈ 0.13GB。",
                   {"size": 9.8, "color": SLATE})], "after": 2},
        {"runs": [("→ PP 的通信量只与「边界数」有关,与层内计算量无关 —— 这就是它能跑在普通网络上的原因。",
                   {"size": 9.8, "color": INK, "bold": True})]},
    ])
    txt(s, ML, 5.72, CW, 0.9,
        "所以选型的第一条直觉:节点内有多卡且 NVLink 可用 → TP 更省延迟;要跨节点、或者机器之间只有普通网络 "
        "→ 必须靠 PP(通常 TP × PP 组合)。反过来,没有 NVLink 的单机也常常 PP 优于 TP。",
        size=10.5, color=SLATE, line=1.3)
    footer(s, 5, "估算口径:按每次集合通信搬运一遍数据计;每 token hidden 8192 × 2B = 16KB")
    notes(s, "表格是结论,右边的图是直觉。重点强调通信量的差别:TP 每层同步,PP 只在边界传一次。"
             "所以没有 NVLink 的机器反而更适合 PP,这是 vLLM 官方文档的明确建议。")


# ----------------------------------------------------------------- slide 6
def s06():
    s = new_slide(TEAL, "03 · 机制  气泡是怎么来的",
                  "一张卡在算,三张卡在等:这就是「流水线气泡」",
                  "把一个批切成 m 个 micro-batch 依次灌入,能填满大部分空转,但首尾总会漏掉一段。")
    txt(s, ML, 1.76, 5.7, 0.26, "① 不切 micro-batch:每批独占 4 个阶段", size=10, color=INK, bold=True)
    cells = {}
    for b in range(2):
        for r in range(4):
            cells[(r, b * 4 + r)] = ("compute", f"B{b + 1}")
    timeline(s, ML, 2.04, 5.75, 4, 8, cells, row_h=0.36, label_w=0.92)
    txt(s, ML, 3.98, 5.7, 0.26, "利用率 ≈ 1/4;每批的延迟 = 4 个阶段之和", size=9, color=RED)

    txt(s, 6.62, 1.76, 5.9, 0.26, "② 切 4 个 micro-batch:阶梯灌入,阶梯流出", size=10, color=INK, bold=True)
    gp = {}
    for c in range(7):
        for r in range(4):
            b = c - r
            if 0 <= b <= 3:
                gp[(r, c)] = ("compute", f"B{b + 1}")
    timeline(s, 6.62, 2.04, 6.07, 4, 7, gp, row_h=0.36, label_w=0.92, cell_w=0.70)
    txt(s, 6.62, 3.98, 5.9, 0.26, "稳态 4 张卡都在算;只有首尾的填充与排空阶段是空的(气泡)", size=9, color=GREEN)

    # formula
    shape(s, ML, 4.38, 6.05, 1.28, fill=INK, radius=0.05)
    txt(s, ML + 0.22, 4.48, 1.8, 0.24, "气泡率", size=9.5, color=AMBER, bold=True)
    txt(s, ML + 1.02, 4.44, 4.2, 0.44, "(p − 1) / (m + p − 1)", size=20, color=WHITE, bold=True,
        font=MONO)
    txt(s, ML + 0.22, 4.94, 5.6, 0.6,
        "p = 阶段数(pp_size),m = micro-batch 数。p 越大、m 越小,空转越严重。\n"
        "p=4: m=4 → 43%,m=8 → 27%,m=32 → 8.6%。",
        size=9, color=RGBColor(0xC7, 0xD2, 0xE0), line=1.25)

    # right cards
    cards = [("训练怎么消泡", "1F1B(一前一后):每个阶段做完一个 micro-batch 的前向,立刻做另一个的反向。"
                              "靠「反向」把前半段空转填上。", TEAL),
             ("推理没有反向", "只有前向,1F1B 用不上。要填满流水线,只能让多个「不同请求的批」同时在飞。", VIOLET)]
    x = 6.92
    for t, d, col in cards:
        shape(s, x, 4.38, 2.84, 1.28, fill=WHITE, line=HAIR, lw=0.9, radius=0.05)
        txt(s, x + 0.14, 4.46, 2.56, 0.24, t, size=9.4, color=col, bold=True)
        txt(s, x + 0.14, 4.72, 2.56, 0.86, d, size=8.2, color=SLATE, line=1.18)
        x += 2.94
    txt(s, ML, 5.86, CW, 0.62,
        "下一页就是 vLLM 的答案:EngineCore 同时挂 pp_size 个批,每个阶段各算各的 —— 稳态下没有空转。",
        size=10.5, color=INK, bold=True, line=1.25)
    footer(s, 6, "气泡率公式:GPipe(Huang et al., 2019);p 为流水线深度,m 为 micro-batch 数")
    notes(s, "气泡要讲透:不是 PP 实现得不好,而是串行本性的必然。左边极限情况 1/4 利用率,"
             "右边 micro-batch 后改善但首尾仍有缺口。公式给出来,方便后面评估 pp 该开多大。")


# ----------------------------------------------------------------- slide 7
def s07():
    s = new_slide(TEAL, "03 · 机制  vLLM 的消泡办法",
                  "让多个「批」同时在飞:每个阶段每时每刻都有活干",
                  "推理不需要等反向传播,只要队列里始终有 pp_size 个批,就能把流水线填满。")
    cells = {}
    for c in range(12):
        for r in range(4):
            if c >= r:
                cells[(r, c)] = ("compute", f"B{c - r + 1}")
    timeline(s, ML, 1.86, CW, 4, 12, cells, row_h=0.40, label_w=1.10, cell_w=0.87,
             caption="t0–t2 为流水线填充期;稳态下每一格都在算,4 个阶段同时处理 4 个不同的批")

    facts = [("max_concurrent_batches = pp_size", "multiproc_executor.py · 注释写得很直白:\n「PP requires PP-size concurrent batches to fill the pipeline」", BLUE),
             ("batch_queue 深度 = pp_size", "engine/core.py · step_with_batch_queue:先填队列、不阻塞,队列满了才回头收最老的结果", TEAL),
             ("结果只从末段回收", "_get_output_rank() = world_size − tp_size:只有末段 TP rank0 的输出会回到 EngineCore", VIOLET)]
    y = 4.42
    for k, v, col in facts:
        shape(s, ML, y, 4.02, 0.74, fill=WHITE, line=HAIR, lw=0.9, radius=0.05)
        txt(s, ML + 0.16, y + 0.09, 3.7, 0.26, k, size=9.6, color=col, bold=True, font=MONO)
        txt(s, ML + 0.16, y + 0.34, 3.72, 0.36, v, size=8, color=SLATE, line=1.12)
        y += 0.80

    # train vs infer
    data = [["", "训练里的 PP", "vLLM 推理的 PP"],
            ["切的是什么", "同一批的 micro-batch", "不同请求组成的批"],
            ["怎么填满", "1F1B(前向 + 反向交错)", "batch queue 里挂 pp_size 个批"],
            ["空转在哪", "首尾的填充/排空", "队列没满时(请求太少)"]]
    table(s, 4.86, 4.42, 7.84, data, [1.42, 3.10, 3.32], row_h=0.42, header_h=0.34,
          size=8.8, header_size=8.8)
    txt(s, 4.86, 6.28, 7.84, 0.52,
        "代价:流水线上同时有 pp_size 个批在飞,同时驻留的 KV Cache 随并发上升;单请求延迟也变高(要过 pp_size 跳)。",
        size=9, color=RED, line=1.15)
    footer(s, 7, "vllm/v1/executor/multiproc_executor.py:474 · vllm/v1/engine/core.py:184, 466")
    notes(s, "这页是 vLLM 相对训练框架最大的不同:它复用 continuous batching,把『批』当成流水线的填充物。"
             "要强调队列深度恰好等于 pp_size 的原因:每个阶段各占一个批。")


# ----------------------------------------------------------------- slide 8
def s08():
    s = new_slide(VIOLET, "04 · 实现  一条命令如何变成 N 个进程",
                  "从 --pipeline-parallel-size 到每张卡上的一个 Worker",
                  "vLLM 里 PP 不是模型代码里的 if,而是一条贯穿「配置 → 执行器 → Worker → 模型」的链路。")
    steps = [("命令行", "--pipeline-parallel-size 2\n(简写 -pp 2)", "engine/arg_utils.py", BLUE, 2.10),
             ("ParallelConfig", "world_size =\npp × tp × pcp", "config/parallel.py", TEAL, 1.95),
             ("Executor", "建进程 / 建 PP 组\n决定队列深度", "v1/executor/*", VIOLET, 2.00),
             ("Worker × N", "每卡一个进程\n等 irecv / 跑本段", "v1/worker/gpu_worker.py", ORANGE, 2.25),
             ("ModelRunner", "吃 intermediate_tensors\n吐下一段", "v1/worker/gpu_model_runner.py", INK, 2.35)]
    x = ML
    for i, (title, sub, path, col, w) in enumerate(steps):
        shape(s, x, 1.78, w, 1.16, fill=WHITE, line=col, lw=1.2, radius=0.06)
        txt(s, x + 0.10, 1.86, w - 0.16, 0.24, title, size=9.8, color=col, bold=True)
        txt(s, x + 0.10, 2.10, w - 0.16, 0.46, sub, size=7.6, color=SLATE, line=1.1)
        txt(s, x + 0.10, 2.62, w - 0.14, 0.24, path, size=6.6, color=MUTED, font=MONO)
        x += w
        if i < 4:
            arrow(s, x + 0.02, 2.36, x + 0.20, 2.36, color=MUTED, lw=1.3)
            x += 0.24

    # rank layout
    txt(s, ML, 3.18, 6.6, 0.26, "rank 布局:TP=2 × PP=2 = 4 张卡(每段 2 卡做 TP)", size=10,
        color=INK, bold=True)
    bands = [(ML, 2.88, "Stage 0 · L0–L19", 0), (3.78, 2.88, "Stage 1 · L20–L39", 1)]
    for bx, bw, text, idx in bands:
        sp = shape(s, bx, 3.48, bw, 0.24, fill=STAGE_SOFT[idx], line=None, radius=0.2)
        label(sp, text, size=7.6, color=STAGE[idx], bold=True)
    for pp in range(2):
        for tp in range(2):
            i = pp * 2 + tp
            gx = ML + pp * 3.16 + tp * 1.50
            sp = shape(s, gx, 3.80, 1.38, 0.76, fill=WHITE, line=STAGE[pp], lw=1.2, radius=0.06)
            label(sp, f"G{i}", size=12, color=STAGE[pp], bold=True)
            txt(s, gx, 4.60, 1.38, 0.2, f"pp{pp} · tp{tp}", size=7.4, color=MUTED,
                align=PP_ALIGN.CENTER)
    # dashed lines: same tp rank across pp -> same PP group
    for tp in range(2):
        x1 = ML + 0 + tp * 1.50 + 0.69
        x2 = ML + 3.16 + tp * 1.50 + 0.69
        ly = 4.90 + tp * 0.22
        arrow(s, x1, ly, x2, ly, color=VIOLET, lw=1.0, dash=MSO_LINE_DASH_STYLE.DASH, head=False)
        arrow(s, x1, ly - 0.07, x1, ly + 0.07, color=VIOLET, lw=1.0, head=False)
        arrow(s, x2, ly - 0.07, x2, ly + 0.07, color=VIOLET, lw=1.0, head=False)
        txt(s, x2 + 0.06, ly - 0.10, 1.0, 0.2, f"PP 组 {tp}", size=7, color=VIOLET)
    txt(s, ML, 5.34, 6.5, 0.24, "虚线 = 同一个 tp 位置、沿 pp 维各取一个 rank,组成一个 PP 组",
        size=7.6, color=VIOLET)

    txt(s, ML, 5.64, 6.5, 0.86,
        "相邻两个 stage 的全局 rank 相差 tp:TP=8 / PP=4 / 32 卡时,stage0 = rank 0–7,"
        "stage1 = rank 8–15 …… 所以一个 stage 天然落在一台 8 卡机上。",
        size=8.6, color=SLATE, line=1.22)

    # right facts
    txt(s, 7.10, 3.18, 5.6, 0.26, "三个容易忽略的点", size=10, color=INK, bold=True)
    pts = [("world_size 只算一个 DP 副本", "pp × tp × pcp;跨 DP 再乘 data_parallel_size", BLUE),
           ("两个后端都支持 PP", "mp(单机多进程)/ ray(跨机);executor.supports_pp = True", TEAL),
           ("PP 与 TP 正交", "all-reduce 只在 TP 组内;PP 只做相邻 stage 的点对点", VIOLET)]
    y = 3.50
    for k, v, col in pts:
        shape(s, 7.10, y, 5.60, 0.72, fill=PANEL, radius=0.05).line.color.rgb = PANEL
        txt(s, 7.26, y + 0.08, 5.2, 0.24, k, size=9.6, color=col, bold=True)
        txt(s, 7.26, y + 0.32, 5.2, 0.3, v, size=8.6, color=SLATE)
        y += 0.80
    code_card(s, 7.10, 6.06, 5.60, 0.88, None,
              ["# config/parallel.py", "world_size = pp * tp * pcp   # 每个 DP 副本"], size=8.0)
    footer(s, 8, "vllm/config/parallel.py:760 · vllm/distributed/parallel_state.py:initialize_model_parallel")
    notes(s, "把链路串起来:命令行只是一个整数,真正的语义在 ParallelConfig 和 executor。"
             "rank 布局要讲清:PP 组不是相邻卡,而是跨 tp 的同一列。")


# ----------------------------------------------------------------- slide 9
def s09():
    s = new_slide(VIOLET, "04 · 实现  DP / TP / PP 同时启用时的卡位",
                  "三种并行同时上:每张卡同时属于 DP、PP、TP 三个组",
                  "例:DP=2、PP=4、TP=2,共 16 卡(2 台 8 卡机)。副本之间复制模型,副本内先切层、再切张量。")
    panels = [(0.80, 0, "副本 0 · 8 卡"), (4.60, 1, "副本 1 · 8 卡")]
    for px, dp, note_text in panels:
        shape(s, px, 2.20, 2.95, 4.04, fill=PANEL, radius=0.04)
        badge = shape(s, px + 0.14, 2.28, 0.72, 0.26, fill=INK, radius=0.3)
        label(badge, f"DP {dp}", size=7.4, color=WHITE, bold=True)
        txt(s, px + 0.94, 2.30, 1.10, 0.22, "一份完整模型", size=7.2, color=SLATE, bold=True)
        txt(s, px + 1.72, 2.30, 1.12, 0.22, f"{note_text} · node {dp}", size=7, color=MUTED,
            align=PP_ALIGN.RIGHT)
        for r in range(4):
            sy = 2.66 + r * 0.90
            shape(s, px + 0.12, sy, 2.71, 0.72, fill=WHITE, line=HAIR, lw=0.8, radius=0.05)
            for t in range(2):
                gx = px + 0.12 + t * 1.40
                rank = dp * 8 + r * 2 + t
                box = shape(s, gx, sy + 0.08, 1.10, 0.56, fill=STAGE_SOFT[r], line=STAGE[r],
                            lw=1.0, radius=0.12)
                rich(s, gx + 0.04, sy + 0.12, 1.02, 0.48, [
                    {"runs": [(f"G{rank}", {"size": 10.5, "color": STAGE[r], "bold": True})],
                     "align": PP_ALIGN.CENTER, "after": 0, "line": 0.95},
                    {"runs": [(f"pp{r} · tp{t}", {"size": 6.4, "color": SLATE})],
                     "align": PP_ALIGN.CENTER, "line": 0.95},
                ], margins=(0.0, 0.0, 0.0, 0.0))
            txt(s, px + 1.26, sy + 0.26, 0.22, 0.2, "⇄", size=9, color=AMBER, bold=True,
                align=PP_ALIGN.CENTER)
            if r < 3:
                for t in range(2):
                    ax = px + 0.12 + t * 1.40 + 0.55
                    arrow(s, ax, sy + 0.72, ax, sy + 0.90, color=MUTED, lw=1.0, head_size="sm")

    # DP relation between replicas
    arrow(s, 3.79, 2.41, 4.56, 2.41, color=INK, lw=1.1, dash=MSO_LINE_DASH_STYLE.DASH, tail=True)
    txt(s, 3.86, 2.12, 0.64, 0.22, "DP", size=7.4, color=INK, bold=True, align=PP_ALIGN.CENTER)

    txt(s, ML, 6.34, 7.55, 0.24,
        "纵向 ↓ = PP(4 段接力,边界传 hidden)· 格内 ⇄ = TP(2 卡 all-reduce)· 两栏 = DP(各一份完整模型)",
        size=7.6, color=SLATE)

    # right: rank formula
    code_card(s, 7.85, 2.24, 4.85, 1.34, "vllm/distributed/parallel_state.py", [
        "rank = (dp × PP + pp) × TP + tp",
        "world_size = DP × PP × TP (× PCP)",
        "DP=2 · PP=4 · TP=2  →  16 卡",
        "G0  = dp0·pp0·tp0  →  (0×4+0)×2+0 = 0",
        "G11 = dp1·pp1·tp1  →  (1×4+1)×2+1 = 11",
    ], size=7.4, hl={4: AMBER})

    data = [["维度", "组内成员", "通信", "物理落点"],
            ["DP", "同 pp、同 tp,不同 dp", "无张量;仅步级同步", "副本之间,可跨机"],
            ["PP", "同 dp、同 tp,不同 pp", "边界一次 P2P", "同机或跨机"],
            ["TP", "同 dp、同 pp,不同 tp", "每层 2 次 all-reduce", "段内,须 NVLink"]]
    table(s, 7.85, 3.72, 4.85, data, [0.66, 1.66, 1.36, 1.17], row_h=0.50, header_h=0.34,
          size=7.8, header_size=7.8)

    shape(s, 7.85, 5.62, 4.85, 1.16, fill=SOFT["amber"], line=AMBER, lw=0.9, radius=0.05)
    txt(s, 8.01, 5.70, 4.5, 0.22, "扩容怎么加", size=8.6, color=AMBER, bold=True)
    rich(s, 8.01, 5.94, 4.55, 0.78, [
        {"runs": [("装不下:先 PP 分段,段内再 TP 切矩阵。", {"size": 7.6, "color": SLATE})],
         "bullet": ("▪", AMBER), "after": 1, "line": 1.05},
        {"runs": [("不够快:加 DP 副本(整模型复制),或加密 PP 段。", {"size": 7.6, "color": SLATE})],
         "bullet": ("▪", AMBER), "after": 1, "line": 1.05},
        {"runs": [("TP 不跨机;没有 NVLink 时优先用 PP 换带宽。", {"size": 7.6, "color": SLATE})],
         "bullet": ("▪", AMBER), "line": 1.05},
    ])
    footer(s, 9, "vllm/distributed/parallel_state.py:initialize_model_parallel(排布 ExternalDP × DP × PP × PCP × TP)")
    notes(s, "这一页把三个维度落到具体卡位。要点:DP 在最外层,一个副本的卡在 rank 空间里连续,"
             "16 卡 / 2 台 8 卡机时一个副本正好一台机器;副本内先按 PP 分段,每段内再按 TP 切。"
             "读图时先看面板(DP),再看行(PP),最后看格内的两个卡(TP)。")


# ----------------------------------------------------------------- slide 10
def s10():
    s = new_slide(VIOLET, "04 · 实现  难题① 层怎么切",
                  "均分 + 余数往前补 + 末段少一层",
                  "32 层、PP=3 时 vLLM 切成 [11, 11, 10]:余数不给末段,因为末段还要放 norm 和 lm_head。")
    # layer bar
    groups = [11, 11, 10]
    x0, y0, cw, gap = ML, 2.12, 0.335, 0.012
    idx = 0
    for g, cnt in enumerate(groups):
        start = idx
        for k in range(cnt):
            shape(s, x0 + idx * (cw + gap), y0, cw, 0.42, fill=STAGE[g], radius=0.12)
            idx += 1
        end = idx - 1
        bx = x0 + start * (cw + gap)
        bw = cnt * (cw + gap) - gap
        shape(s, bx, y0 - 0.30, bw, 0.02, fill=STAGE[g], kind=MSO_SHAPE.RECTANGLE).line.fill.background()
        txt(s, bx, y0 - 0.52, bw, 0.22, f"Stage {g} · {cnt} 层", size=9.5, color=STAGE[g], bold=True,
            align=PP_ALIGN.CENTER)
        txt(s, bx, y0 + 0.46, bw, 0.2, f"L{start}–L{end}", size=8, color=MUTED, align=PP_ALIGN.CENTER)
    shape(s, x0 - 0.05, y0 - 0.02, 32 * (cw + gap) + 0.02, 0.46, fill=None, line=HAIR, lw=1.0,
          kind=MSO_SHAPE.RECTANGLE)

    txt(s, ML, 2.80, 6.5, 0.26, "32 层 ÷ 3 段:每段先给 10 层,余 2 层从倒数第二段往前补", size=9.5, color=SLATE)

    # rules
    txt(s, ML, 3.22, 6.1, 0.26, "默认规则(get_pp_indices)", size=10, color=INK, bold=True)
    rules = [("① 均分", "layers_per_partition = 总层数 // pp_size", BLUE),
             ("② 余数往前补", "余 r 层,从倒数第二段开始往前每段 +1;末段不补", TEAL),
             ("③ 末段最少", "末段还要放 RMSNorm + LM Head,首段要放 Embedding", VIOLET),
             ("④ 可手动覆盖", 'VLLM_PP_LAYER_PARTITION="8,8,8,8"(个数=pp,总和=总层数)', ORANGE)]
    y = 3.52
    for k, v, col in rules:
        txt(s, ML, y + 0.04, 1.30, 0.24, k, size=9.4, color=col, bold=True)
        txt(s, ML + 1.34, y + 0.03, 4.70, 0.44, v, size=9, color=SLATE, line=1.12)
        y += 0.50

    # missing layer explain
    shape(s, ML, 5.62, 6.05, 1.08, fill=SOFT["violet"], line=VIOLET, lw=0.9, radius=0.05)
    txt(s, ML + 0.18, 5.70, 2.4, 0.24, "不在本段的层怎么办?", size=9.4, color=VIOLET, bold=True)
    rich(s, ML + 0.18, 5.94, 5.7, 0.7, [
        {"runs": [("PPMissingLayer 占位:", {"size": 8.8, "color": INK, "bold": True}),
                  ("结构上占个位置,但不分配权重、不参与计算;make_layers 只为 [start, end) 真正建层。",
                   {"size": 8.8, "color": SLATE})], "line": 1.15, "after": 1},
        {"runs": [("权重加载也跳过:", {"size": 8.8, "color": INK, "bold": True}),
                  ("is_pp_missing_parameter() 让加载器跳过其他段的参数,只为本段分配显存。",
                   {"size": 8.8, "color": SLATE})], "line": 1.15},
    ])

    # code
    code_card(s, 6.95, 3.22, 5.78, 2.00, "vllm/distributed/utils.py · get_pp_indices", [
        "partitions = [L // pp] * pp",
        "if r := L % pp:",
        "    for i in range(2, r + 2):",
        "        partitions[-i] += 1   # 从倒数第二段往前",
        "start = sum(partitions[:pp_rank])",
        "end   = start + partitions[pp_rank]",
        "return start, end",
    ], size=8.8, hl={3: AMBER})
    txt(s, 6.95, 5.34, 5.78, 1.30,
        "注意:pp_size 必须 ≤ 层数;层数不能整除时不会报错,而是自动均摊(日志会打印 "
        "Hidden layers were unevenly partitioned)。想手工平衡首末段的显存,就直接设环境变量。",
        size=8.6, color=SLATE, line=1.2)
    footer(s, 10, "vllm/distributed/utils.py:95 · vllm/model_executor/models/utils.py:607,620")
    notes(s, "切层是最容易被忽略却最影响显存平衡的地方。重点讲清余数分配方向,以及首末段额外的 embedding/lm_head。"
             "PPMissingLayer 是理解 vLLM 模型代码的关键技巧。")


# ----------------------------------------------------------------- slide 11
def s11():
    s = new_slide(VIOLET, "04 · 实现  难题② 段间怎么传数据",
                  "只传两个张量,而且发送之后不等",
                  "每个 stage 算完自己那段,把 hidden_states 和 residual 交给下一段;元数据走 CPU,张量走 NCCL。")
    # protocol diagram
    shape(s, ML, 1.84, 2.05, 0.92, fill=STAGE_SOFT[0], line=STAGE[0], lw=1.1, radius=0.06)
    txt(s, ML + 0.14, 1.94, 1.8, 0.26, "Stage i", size=11, color=STAGE[0], bold=True)
    txt(s, ML + 0.14, 2.22, 1.82, 0.44, "算完本段\n产出 IntermediateTensors", size=8, color=SLATE, line=1.12)
    shape(s, 3.95, 1.84, 2.05, 0.92, fill=STAGE_SOFT[1], line=STAGE[1], lw=1.1, radius=0.06)
    txt(s, 4.09, 1.94, 1.8, 0.26, "Stage i+1", size=11, color=STAGE[1], bold=True)
    txt(s, 4.09, 2.22, 1.82, 0.44, "收齐后\n接着算下一段", size=8, color=SLATE, line=1.12)
    arrow(s, 2.74, 2.08, 3.90, 2.08, color=MUTED, lw=1.4, dash=MSO_LINE_DASH_STYLE.DASH)
    arrow(s, 2.74, 2.52, 3.90, 2.52, color=BLUE, lw=2.2)
    txt(s, 2.76, 1.86, 1.14, 0.2, "① 元数据 / gloo(CPU)", size=7, color=MUTED, bold=True)
    txt(s, 2.76, 2.58, 1.14, 0.2, "② 张量 / NCCL(GPU)", size=7, color=BLUE, bold=True)
    txt(s, ML, 2.86, 5.9, 0.46,
        "① 先发元数据:有几个张量、形状与 dtype  →  ② 再传张量本体:isend / irecv(NCCL 点对点)",
        size=8.4, color=SLATE, line=1.15)

    # overlap timeline
    txt(s, ML, 3.36, 6.1, 0.26, "为什么说「发送之后不等」:send 立刻返回,通信被下一轮计算盖住", size=9.6,
        color=INK, bold=True)
    bars = [("recv", 0.0, 0.9, SOFT["amber"], AMBER),
            ("compute", 0.9, 2.6, STAGE[0], STAGE[0]),
            ("send", 2.6, 3.5, SOFT["violet"], VIOLET)]
    bx, by, scale = ML + 0.78, 3.68, 1.24
    for name, st, en, fill, col in bars:
        txt(s, ML, by + 0.04, 0.72, 0.2, name, size=8.2, color=col, bold=True, align=PP_ALIGN.RIGHT)
        shape(s, bx + st * scale, by, (en - st) * scale, 0.26, fill=fill, line=None, radius=0.15)
        by += 0.34
    arrow(s, bx + 3.5 * scale, 3.62, bx + 3.5 * scale, 4.68, color=MUTED, lw=1.0,
          dash=MSO_LINE_DASH_STYLE.DASH)
    txt(s, 4.30, 4.70, 1.40, 0.22, "下一轮开头才 wait()", size=7.2, color=MUTED, align=PP_ALIGN.RIGHT)

    # right: three points
    pts = [("传什么", "{hidden_states, residual} 两个张量;不同模型键可能更多,但都是「边界处的中间结果」", BLUE),
           ("怎么省", "TP>1 时每个 rank 只发 1/tp 切片,接收方在自己的 TP 组内 all-gather 还原", TEAL),
           ("怎么懒", "AsyncIntermediateTensors:模型第一层真正用到它时才同步,不阻塞 host", VIOLET)]
    y = 1.84
    for k, v, col in pts:
        shape(s, 6.72, y, 5.98, 0.90, fill=WHITE, line=HAIR, lw=0.9, radius=0.05)
        txt(s, 6.88, y + 0.10, 1.2, 0.24, k, size=9.6, color=col, bold=True)
        txt(s, 8.05, y + 0.09, 4.55, 0.72, v, size=8.6, color=SLATE, line=1.15)
        y += 0.98

    code_card(s, ML, 5.06, CW, 1.62, "vllm/v1/worker/gpu_worker.py · execute_model(节选)", [
        "if forward_pass and not get_pp_group().is_first_rank:",
        "    tensor_dict, handles, post = get_pp_group().irecv_tensor_dict(",
        "        all_gather_group=get_tp_group(), all_gather_tensors=...)",
        "    intermediate_tensors = AsyncIntermediateTensors(tensor_dict, handles, post)",
        "output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)",
        "self._pp_send_work = get_pp_group().isend_tensor_dict(output.tensors, ...)  # 不等",
    ], size=8.4, hl={4: AMBER, 5: AMBER})
    footer(s, 11, "vllm/distributed/parallel_state.py:821-1038 · vllm/v1/worker/gpu_worker.py:782")
    notes(s, "两个层次:协议(元数据+张量)和时序(异步)。学生最容易忽略的是『发送不阻塞』,"
             "它让 PP 的通信几乎不占关键路径。")


# ----------------------------------------------------------------- slide 12
def s12():
    s = new_slide(VIOLET, "04 · 实现  难题③ 采样结果怎么回到第一段",
                  "只有末段能采样,但每一段都要知道新 token",
                  "首段要拼下一次的输入,每段都要更新自己那几层的 KV —— 于是必须把采样结果广播回去。")
    # fan-out diagram (broadcast bus)
    last = shape(s, 2.00, 1.92, 3.10, 0.54, fill=STAGE_SOFT[3], line=STAGE[3], lw=1.1, radius=0.08)
    label(last, "Stage 3(末段)· 算 logits → sample", size=8.6, color=STAGE[3], bold=True)
    txt(s, 2.00, 2.50, 3.55, 0.22, "_pp_broadcast_prev_sampled_token_ids · gpu_model_runner.py",
        size=6.2, color=MUTED, font=MONO)
    # bus: from last stage left edge down, then into each stage
    bus_x = 1.30
    arrow(s, 1.96, 2.19, bus_x + 0.02, 2.19, color=STAGE[3], lw=1.4)
    shape(s, bus_x - 0.012, 2.19, 0.024, 2.02, fill=STAGE[3], kind=MSO_SHAPE.RECTANGLE).line.fill.background()
    stages = [2.86, 3.44, 4.02]
    for i, yy in enumerate(stages):
        box = shape(s, 1.90, yy, 3.20, 0.40, fill=STAGE_SOFT[i], line=STAGE[i], lw=0.9, radius=0.1)
        label(box, f"Stage {i} · 收广播 → 写回 input batch", size=8, color=STAGE[i], bold=True)
        arrow(s, bus_x, yy + 0.20, 1.88, yy + 0.20, color=STAGE[3], lw=1.3)

    # why
    shape(s, ML, 4.62, 5.55, 1.00, fill=SOFT["amber"], line=AMBER, lw=0.9, radius=0.05)
    txt(s, ML + 0.18, 4.70, 3.2, 0.24, "为什么每张卡都要 token?", size=9.4, color=AMBER, bold=True)
    rich(s, ML + 0.18, 4.96, 5.2, 0.62, [
        {"runs": [("每段只为自己那几层存 KV Cache,但采样、停止判断、位置推进是全请求共享的状态。",
                   {"size": 8.8, "color": SLATE})], "line": 1.16, "after": 2},
        {"runs": [("所以 PP 下所有 stage 都在维护同一份请求账本,只是各管各的 KV。",
                   {"size": 8.8, "color": INK, "bold": True})], "line": 1.16},
    ])

    # two paths
    txt(s, 6.55, 1.84, 6.1, 0.26, "两条回传路径,是否开异步调度(async scheduling)决定走哪条", size=9.6,
        color=INK, bold=True)
    shape(s, 6.55, 2.14, 6.15, 1.30, fill=WHITE, line=HAIR, lw=0.9, radius=0.05)
    txt(s, 6.72, 2.22, 5.8, 0.24, "路径 A · GPU 广播(PP 组内,不经 CPU)", size=9.4, color=BLUE, bold=True)
    txt(s, 6.72, 2.46, 5.78, 0.62,
        "末段采样后把 int32 的 prev_sampled_token_ids 在 PP 组内广播;其余段在下一步 execute_model "
        "开头接收,直接写进 input batch。",
        size=8.4, color=SLATE, line=1.18)
    txt(s, 6.72, 3.06, 5.78, 0.32,
        "_pp_receive_prev_sampled_token_ids_to_input_batch()", size=6.6, color=MUTED, font=MONO)

    shape(s, 6.55, 3.54, 6.15, 1.36, fill=WHITE, line=HAIR, lw=0.9, radius=0.05)
    txt(s, 6.72, 3.62, 5.8, 0.24, "路径 B · 调度器回填(仅未开异步调度)", size=9.4, color=TEAL,
        bold=True)
    txt(s, 6.72, 3.88, 5.78, 0.96,
        "scheduler 把本步的 new_token_ids 塞进 CachedRequestData 一起下发。源码注释解释了原因:"
        "「there's no direct communication between the first-stage worker and the last-stage worker」。",
        size=8.4, color=SLATE, line=1.18)
    txt(s, 6.55, 5.02, 6.15, 0.46,
        "开了 async scheduling 后由 GPU 广播直接喂 input batch,调度器省略这段负载;未开时两条并列。",
        size=8, color=MUTED, line=1.16)

    code_card(s, ML, 5.74, CW, 1.12, "vllm/v1/core/sched/scheduler.py:1019-1027", [
        "if self.use_pp and not self.scheduler_config.async_scheduling:",
        "    # When using PP, the scheduler sends the sampled tokens back, because",
        "    # there's no direct communication between the first-stage worker",
        "    # and the last-stage worker.",
    ], size=8.0, hl={1: AMBER, 2: AMBER, 3: AMBER})
    footer(s, 12, "vllm/v1/worker/gpu_model_runner.py:4481, 4498 · gpu/pp_utils.py 为 V2 runner 的等价实现")
    notes(s, "这页回答『采样在末段,结果怎么回去』。默认 runner(gpu_model_runner.py)用 PP 组广播 prev_sampled_token_ids;"
             "未开 async scheduling 时调度器还会回填 new_token_ids。V2 runner(VLLM_USE_V2_MODEL_RUNNER=1)对应 "
             "gpu/pp_utils.py 的 pp_broadcast / pp_receive。")


# ----------------------------------------------------------------- slide 13
def s13():
    s = new_slide(VIOLET, "04 · 实现  一次 forward 的四步",
                  "所有 stage 跑同一份代码,靠 is_first / is_last 分叉",
                  "worker.execute_model 每步就四件事:等上一轮发送、收、算、发。首段少收,末段少发。")
    steps = [("①", "等上一轮发送", "上一轮的发送还没落账,先把 handle(异步句柄)等完:保证 NCCL 顺序、不覆写待发缓冲。", BLUE),
             ("②", "非首段:异步收", "irecv_tensor_dict 把张量挂到 NCCL 流上,包成 AsyncIntermediateTensors。", TEAL),
             ("③", "跑本段的层", "model_runner 用收到的 hidden/residual 作为输入,只跑 [start, end) 这段。", VIOLET),
             ("④", "非末段:异步发", "isend_tensor_dict 立刻返回 handle,不阻塞,下一轮开头再 wait。", ORANGE)]
    x = ML
    for num, title, desc, col in steps:
        shape(s, x, 1.72, 2.96, 1.58, fill=WHITE, line=HAIR, lw=0.9, radius=0.05)
        txt(s, x + 0.16, 1.82, 0.4, 0.3, num, size=17, color=col, bold=True)
        txt(s, x + 0.62, 1.86, 2.2, 0.26, title, size=10.5, color=INK, bold=True)
        txt(s, x + 0.16, 2.26, 2.66, 0.96, desc, size=8.6, color=SLATE, line=1.2)
        x += 3.05

    code_card(s, ML, 3.50, 7.92, 2.86, "vllm/v1/worker/gpu_worker.py · GpuWorker.execute_model", [
        "def execute_model(self, scheduler_output):",
        "    # ① 等上一轮的异步发送落账",
        "    for handle in self._pp_send_work:",
        "        handle.wait()",
        "    # ② 非首段:从上一段异步收",
        "    if forward_pass and not get_pp_group().is_first_rank:",
        "        tensor_dict, handles, post = get_pp_group().irecv_tensor_dict(...)",
        "        intermediate_tensors = AsyncIntermediateTensors(tensor_dict, handles, post)",
        "    # ③ 跑本段的层(可与收重叠)",
        "    output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)",
        "    if isinstance(output, (ModelRunnerOutput, AsyncModelRunnerOutput, NoneType)):",
        "        return output          # 末段 / 池化 / 空批:不需要转发",
        "    # ④ 非末段:异步发给下一段,返回 None",
        "    self._pp_send_work = get_pp_group().isend_tensor_dict(output.tensors, ...)",
    ], size=8.3, hl={1: RGBColor(0x94, 0xA3, 0xB8), 4: RGBColor(0x94, 0xA3, 0xB8),
                     8: RGBColor(0x94, 0xA3, 0xB8), 12: RGBColor(0x94, 0xA3, 0xB8)})

    # right: branch table
    data = [["阶段", "② 收", "③ 算", "④ 发", "返回"],
            ["首段", "—", "✓", "✓", "None"],
            ["中间段", "✓", "✓", "✓", "None"],
            ["末段", "✓", "✓", "—", "采样 → 输出"]]
    table(s, 8.80, 3.42, 3.90, data, [0.92, 0.62, 0.62, 0.62, 1.12], row_h=0.44, header_h=0.36,
          size=8.4, header_size=8.4,
          aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER, PP_ALIGN.CENTER, PP_ALIGN.LEFT])
    txt(s, 8.80, 5.34, 3.90, 1.06,
        "末段的 model_runner 看到的是 hidden_states,交给 sampler(采样器)出 token;"
        "其他段看到的是 IntermediateTensors,直接原样返回给 worker 去发。",
        size=8.4, color=SLATE, line=1.2)
    txt(s, ML, 6.62, CW, 0.3,
        "注意:同一份代码在 4 种角色下执行 —— 这就是为什么读 vLLM 的 PP 只需要读一个 execute_model。",
        size=9.4, color=INK, bold=True)
    footer(s, 13, "vllm/v1/worker/gpu_worker.py:782-871 · gpu_model_runner.py:4234, 4281")
    notes(s, "源码页,四步对应四个 if。强调首段/末段的分叉,以及『末段采样、其他段原样返回』。")


# ----------------------------------------------------------------- slide 14
def s14():
    s = new_slide(ORANGE, "04 · 实现  这笔账怎么算",
                  "PP 省的是显存,不是算力",
                  "权重和 KV Cache 都按层摊到各段;但总计算量不变,单请求还要多走 pp_size 跳。")
    data = [["显存项", "单卡部署", "PP=4 每卡", "说明"],
            ["模型权重", "100%", "≈ 25%", "只加载自己那段层;其余是 PPMissingLayer"],
            ["KV Cache", "全部层", "≈ 25%", "只为自己的注意力层分配 block"],
            ["激活 / 隐状态", "整模型", "本段 + 收发缓冲", "缓冲按 max_num_batched_tokens 预留"],
            ["Embedding / LM Head", "各 1 份", "只在首段 / 末段", "末段通常最紧,可手动少切一层"]]
    table(s, ML, 1.80, 7.85, data, [1.92, 1.72, 1.72, 2.49], row_h=0.56, header_h=0.42,
          size=9.6, header_size=9.8)

    nums = [("140 GB", "70B 模型 fp16 权重", "一张 80GB 卡装不下 → 必须切", RED),
            ("320 KB", "70B 每 token 的 KV(全模型)", "PP=4 后每卡 ≈ 80KB/token", BLUE),
            ("128 MB", "8K token 批过一次边界的 hidden", "对比 TP 每层 all-reduce 的 GB 级", GREEN)]
    y = 1.80
    for big, cap, sub, col in nums:
        shape(s, 8.72, y, 3.98, 0.94, fill=PANEL, radius=0.05).line.color.rgb = PANEL
        txt(s, 8.88, y + 0.08, 2.0, 0.4, big, size=18, color=col, bold=True)
        txt(s, 10.55, y + 0.14, 2.0, 0.24, cap, size=8, color=SLATE, align=PP_ALIGN.RIGHT)
        txt(s, 8.88, y + 0.52, 3.7, 0.3, sub, size=8.2, color=MUTED)
        y += 1.02

    txt(s, ML, 4.86, 7.85, 0.3, "所以 PP 的收益 / 代价要分开看", size=11, color=INK, bold=True)
    gains = ["容量:单模型可跨机,突破单卡显存限制", "规模:卡数增加时可承载更大的模型(吞吐靠 DP)", "通信:只需点对点,普通网络可承受"]
    costs = ["延迟:每 token 多 pp_size 跳,首 token 与每 token 延迟(TTFT/TPOT)变高", "气泡:请求少时流水线填不满", "约束:模型要支持 SupportsPP"]
    shape(s, ML, 5.20, 5.95, 1.12, fill=SOFT["green"], line=GREEN, lw=0.9, radius=0.05)
    txt(s, ML + 0.18, 5.28, 2.0, 0.24, "收益", size=9.6, color=GREEN, bold=True)
    rich(s, ML + 0.18, 5.54, 5.5, 0.72,
         [{"runs": [(g, {"size": 9, "color": SLATE})], "bullet": ("▪", GREEN), "after": 1, "line": 1.1}
          for g in gains])
    shape(s, 6.62, 5.20, 6.08, 1.12, fill=SOFT["red"], line=RED, lw=0.9, radius=0.05)
    txt(s, 6.80, 5.28, 2.0, 0.24, "代价", size=9.6, color=RED, bold=True)
    rich(s, 6.80, 5.54, 5.65, 0.72,
         [{"runs": [(c, {"size": 9, "color": SLATE})], "bullet": ("▪", RED), "after": 1, "line": 1.1}
          for c in costs])
    footer(s, 14, "估算:KV 按 Llama-3-70B(80 层 / 8 KV heads / head_dim 128 / fp16);hidden 8192")
    notes(s, "把收益和代价摆在一页,避免『PP 一定更快』的误解。数字都是可复算的估算,讲的时候给公式。")


# ----------------------------------------------------------------- slide 15
def s15():
    s = new_slide(ORANGE, "05 · 实战  怎么开、怎么选",
                  "三条命令覆盖 90% 的场景",
                  "单机用 mp(多进程),跨机用 ray 或 mp 多节点;选型只记一句话:节点内 TP,节点间 PP。")
    code_card(s, ML, 1.76, 7.62, 1.10, "① 单机 8 卡:TP=4 × PP=2,都在机器内", [
        "vllm serve <model> --tensor-parallel-size 4 --pipeline-parallel-size 2",
        "python -m vllm.entrypoints.openai.api_server ...   # 等价写法",
        "LLM(model, tensor_parallel_size=4, pipeline_parallel_size=2)   # 离线",
    ], size=8.4)
    code_card(s, ML, 2.94, 7.62, 1.16, "② 2 机 × 8 卡:每机 TP=8,跨机 PP=2(ray 集群)", [
        "# 两台机器在同一 ray 集群里,一条命令即可",
        "vllm serve <model> --tensor-parallel-size 8 --pipeline-parallel-size 2 \\",
        "    --distributed-executor-backend ray",
    ], size=8.4)
    code_card(s, ML, 4.18, 7.62, 1.10, "③ 手动指定每段层数(总和必须 = 总层数)", [
        "export VLLM_PP_LAYER_PARTITION=\"20,20,20,20\"   # 4 段,共 80 层",
        "vllm serve <model> -tp 2 -pp 4",
    ], size=8.4)
    txt(s, ML, 5.42, 7.62, 1.20,
        "多节点用 mp 后端也可以:两台机器分别加 --nnodes 2 --node-rank 0/1 --master-addr <IP>,"
        "从节点再加 --headless。PP=卡数且层数不整除时,直接用环境变量平衡各段。",
        size=8.4, color=SLATE, line=1.2)

    # right: selection
    txt(s, 8.48, 1.76, 4.2, 0.26, "官方选型(以 docs 为准)", size=10, color=INK, bold=True)
    sel = [("单卡放得下", "不要并行", MUTED),
           ("单卡放不下、单机放得下", "节点内 TP", BLUE),
           ("单机放不下", "TP(机内)× PP(机间)", TEAL),
           ("卡没有 NVLink(如 L40S)", "优先 PP,而不是 TP", GREEN),
           ("GPU 数与层数不整除", "PP 支持不均匀切分", ORANGE)]
    y = 2.06
    for a, b, col in sel:
        txt(s, 8.48, y + 0.03, 1.75, 0.5, a, size=8.2, color=SLATE, line=1.1)
        chip(s, 10.28, y - 0.01, 2.42, 0.44, b, fill=SOFT["grey"] if col == MUTED else PANEL2,
             color=col, size=8, line=None)
        y += 0.54

    shape(s, 8.48, 4.82, 4.22, 0.98, fill=SOFT["amber"], line=AMBER, lw=0.9, radius=0.05)
    txt(s, 8.64, 4.90, 3.9, 0.24, "进阶开关(--enable-dbo)", size=9, color=AMBER, bold=True)
    txt(s, 8.64, 5.14, 3.95, 0.60,
        "把一个批再切成 2 个 micro-batch,让集合通信(如 MoE all-to-all)与计算重叠;"
        "阈值 --dbo-decode-token-threshold / --dbo-prefill-token-threshold 控制何时启用。",
        size=7.6, color=SLATE, line=1.12)

    data = [["参数", "作用"],
            ["-pp / --pipeline-parallel-size", "段数 = 进程数 ÷ TP"],
            ["--distributed-executor-backend", "mp(单机)/ ray(跨机)"],
            ["VLLM_PP_LAYER_PARTITION", "手工指定每段层数"]]
    table(s, 8.48, 5.94, 4.22, data, [2.02, 2.20], row_h=0.24, header_h=0.24, size=6.9,
          header_size=7.1)
    footer(s, 15, "docs/serving/parallelism_scaling.md · vllm/engine/arg_utils.py")
    notes(s, "三条命令按场景背下来。右边是官方文档的四条选型建议,最后一条(L40S 无 NVLink)很重要,"
             "很多人的直觉是反的 —— 没 NVLink 时 TP 反而慢。")


# ----------------------------------------------------------------- slide 16
def s16():
    s = new_slide(ORANGE, "05 · 实战  踩坑清单",
                  "7 个硬限制 + 3 步排错",
                  "PP 不是开个开关就完事:模型要支持、特性要兼容、每段的批必须完全一致,否则直接挂住。")
    left = [("模型必须实现 SupportsPP", "启动即校验 registry.is_pp_supported_model();冷门模型直接 NotImplementedError"),
            ("EAGLE3 投机采样 × PP", "EAGLE3 需要目标模型多个中间层的 aux hidden states,PP 把层切开了 → 启动报错"),
            ("Elastic EP × PP", "ParallelConfig 初始化直接报错;普通 EP + PP 不受影响"),
            ("--enable-return-routed-experts", "源码明确标注「PP > 1 未验证」,会拒绝启动"),
            ("多模态 torch_shm IPC", "mm_tensor_ipc='torch_shm' 与 pp>1 互斥"),
            ("pp_size > 层数", "每段至少要有一层;pp 比层数还大时切不出有效阶段"),
            ("同一 DP 组必须同时 generate", "否则 PP 组内点对点通信对不上,直接死锁")]
    txt(s, ML, 1.76, 6.0, 0.26, "先避坑(全部是启动期或通信期的硬限制)", size=10, color=INK, bold=True)
    y = 2.04
    for k, v in left[:4]:
        txt(s, ML, y + 0.02, 0.16, 0.22, "▪", size=10, color=RED, bold=True)
        txt(s, ML + 0.18, y, 5.7, 0.24, k, size=9.2, color=INK, bold=True)
        txt(s, ML + 0.18, y + 0.24, 5.72, 0.34, v, size=8.2, color=SLATE, line=1.1)
        y += 0.68
    y = 4.76
    for k, v in left[4:]:
        txt(s, ML, y + 0.02, 0.16, 0.22, "▪", size=10, color=RED, bold=True)
        txt(s, ML + 0.18, y, 5.7, 0.24, k, size=9.2, color=INK, bold=True)
        txt(s, ML + 0.18, y + 0.24, 5.72, 0.34, v, size=8.2, color=SLATE, line=1.1)
        y += 0.66

    txt(s, 6.62, 1.76, 6.0, 0.26, "排错三步", size=10, color=INK, bold=True)
    steps = [("① 看日志:切层与角色", "启动日志会打印 unevenly partitioned;每段日志的进程名带 _PP{rank},先确认谁是首/末段。", BLUE),
             ("② 看批:每段是否一致", "PP 要求所有 stage 拿到同一份 SchedulerOutput。某段多跑/少跑一个批 → NCCL 等不到对端。", TEAL),
             ("③ 看通信:NCCL P2P", "NCCL_DEBUG=INFO 确认走的是 NVLink/NET/IB;跨机时若看到 NET/Socket,先修网络。", VIOLET)]
    y = 2.06
    for k, v, col in steps:
        shape(s, 6.62, y, 6.08, 0.96, fill=WHITE, line=HAIR, lw=0.9, radius=0.05)
        txt(s, 6.78, y + 0.09, 5.6, 0.24, k, size=9.4, color=col, bold=True)
        txt(s, 6.78, y + 0.35, 5.72, 0.56, v, size=8.4, color=SLATE, line=1.15)
        y += 1.04

    shape(s, 6.62, 5.32, 6.08, 1.28, fill=SOFT["amber"], line=AMBER, lw=0.9, radius=0.05)
    txt(s, 6.78, 5.40, 4.0, 0.24, "性能不升反降时,先问三个问题", size=9.2, color=AMBER, bold=True)
    rich(s, 6.78, 5.66, 5.7, 0.86, [
        {"runs": [("并发请求数是否 ≥ pp_size?不够就先加压测并发。", {"size": 8.4, "color": SLATE})],
         "bullet": ("▪", AMBER), "after": 1, "line": 1.1},
        {"runs": [("首/末段是否显存吃紧?用 VLLM_PP_LAYER_PARTITION 少切一层。", {"size": 8.4, "color": SLATE})],
         "bullet": ("▪", AMBER), "after": 1, "line": 1.1},
        {"runs": [("跨机链路是否走了 IB?没有 IB 的 PP 也可能变成瓶颈。", {"size": 8.4, "color": SLATE})],
         "bullet": ("▪", AMBER), "line": 1.1},
    ])
    footer(s, 16, "vllm/config/model.py:1153 · vllm/config/parallel.py:775 · model_runner.py:180 · vllm/config/vllm.py:1952")
    notes(s, "坑分三类:能力(不支持)、一致性(死锁)、性能(填不满)。排错三步按日志→批→通信的顺序。"
             "特别提醒 DP 组必须同步 generate 这条,生产上很容易踩。")


# ----------------------------------------------------------------- slide 17
def s17():
    s = new_slide(INK, "收尾 · 三件事与源码索引",
                  "切层、接力、填满:PP 就这三件事",
                  "如果只记三个数字:pp_size、max_concurrent_batches = pp_size、(p−1)/(m+p−1)。")
    cards = [("切", "按层切段", "get_pp_indices 均分,余数往前补,末段少一层", BLUE, "P10"),
             ("传", "边界传 hidden", "只传 {hidden_states, residual};元数据走 gloo、张量走 NCCL", TEAL, "P11"),
             ("填", "多批填满", "batch_queue 挂 pp_size 个批,每段都有活干", VIOLET, "P7")]
    x = ML
    for ch, title, desc, col, ref in cards:
        shape(s, x, 1.68, 3.93, 0.94, fill=WHITE, line=HAIR, lw=0.9, radius=0.05)
        txt(s, x + 0.14, 1.76, 0.50, 0.38, ch, size=16, color=col, bold=True)
        txt(s, x + 0.60, 1.78, 2.30, 0.24, title, size=9.8, color=INK, bold=True)
        txt(s, x + 0.14, 2.14, 3.60, 0.40, desc, size=7.8, color=SLATE, line=1.12)
        txt(s, x + 3.24, 2.14, 0.56, 0.20, ref, size=7, color=col, bold=True, align=PP_ALIGN.RIGHT)
        x += 4.06

    data = [["源码位置", "管什么(按一次 PP 前向的顺序读)", "对应页"],
            ["config/parallel.py",
             "定义 pipeline_parallel_size,并由它算出 world_size = pp × tp × pcp;同时校验哪些特性与 PP 互斥(如 Elastic EP、torch_shm)。",
             "P8"],
            ["distributed/parallel_state.py",
             "按 ExternalDP × DP × PP × PCP × TP 建进程组;提供 get_pp_group() 与 send/recv_tensor_dict 点对点收发。",
             "P8 – P9"],
            ["distributed/utils.py",
             "get_pp_indices():把总层数按 pp_size 均分,余数从倒数第二段往前补,定出每段的 [start, end)。",
             "P10"],
            ["model_executor/models/utils.py",
             "make_layers() 只为本段真正建层,其余位置放 PPMissingLayer 占位;is_pp_missing_parameter() 判断参数是否属于本段。",
             "P10"],
            ["v1/worker/gpu_worker.py",
             "execute_model() 四步:等上一轮发送 →(非首段)irecv 收 → 跑本段 →(非末段)isend 发,收发都不阻塞。",
             "P11 – P13"],
            ["v1/worker/gpu_model_runner.py",
             "默认 runner:非首段把收到的 hidden/residual 拼进输入;末段采样后把 token 经 PP 组广播给其余段。",
             "P12"],
            ["v1/engine/core.py",
             "EngineCore 用 batch_queue 挂住 pp_size 个批:先填队列、不阻塞,队列满了才回收最老的结果。",
             "P7"],
            ["v1/executor/multiproc_executor.py",
             "max_concurrent_batches = pp_size(队列深度);并确定只从末段 TP0 回收 ModelRunnerOutput。",
             "P7 – P8"]]
    table(s, ML, 2.76, CW, data, [3.05, 7.00, 2.04], row_h=0.42, header_h=0.34, size=8.2,
          header_size=8.4)

    txt(s, ML, 6.54, CW, 0.24,
        "读法:先看 P10 的切层,再看 P11–P13 的收发与采样,最后看 P7 的队列 —— 这几段代码连起来,就是一次完整的 PP 前向。",
        size=8.6, color=MUTED)
    footer(s, 17, "配图与脚本:assets/vllm-novidia/build_pp_intro_deck.py · 延伸阅读:vllm-main-PP流水线并行详解.pptx")
    notes(s, "收尾:三件事 + 源码索引。每条都写清了这个文件到底做什么、对应前面哪一页,"
             "听众可以照着表逐行去仓库里读。如果只做一件事:去读 gpu_worker.execute_model。")


# ----------------------------------------------------------------- slide 18 (references)
def s18():
    s = new_slide(INK, "参考 · 文档与论文",
                  "参考文档与源码",
                  "以下链接对应正文中的结论;具体源码位置见 P17 的索引表。")
    doc = "https://docs.vllm.ai/en/latest/serving/"
    data = [["类别", "名称与用途", "链接"],
            ["官方文档", "Parallelism and Scaling —— 单机/多机选型:何时用 TP、何时 TP×PP,以及 Ray 与多进程部署",
             [(doc + "parallelism_scaling/", {"link": doc + "parallelism_scaling/", "font": MONO,
                                              "size": 7.3, "color": BLUE})]],
            ["官方文档", "Data Parallel Deployment —— DP 副本、负载均衡与部署形态",
             [(doc + "data_parallel_deployment/", {"link": doc + "data_parallel_deployment/",
                                                   "font": MONO, "size": 7.3, "color": BLUE})]],
            ["官方文档", "Troubleshooting distributed deployments —— 分布式部署的排错清单",
             [(doc + "distributed_troubleshooting/", {"link": doc + "distributed_troubleshooting/",
                                                      "font": MONO, "size": 7.3, "color": BLUE})]],
            ["源码", "vLLM 仓库(main 分支)—— 本文全部文件路径与行号的出处",
             [("https://github.com/vllm-project/vllm", {"link": "https://github.com/vllm-project/vllm",
                                                        "font": MONO, "size": 7.3, "color": TEAL})]],
            ["论文", "GPipe(2019)—— micro-batch 流水线与气泡率公式的出处",
             [("https://arxiv.org/abs/1811.06965", {"link": "https://arxiv.org/abs/1811.06965",
                                                   "font": MONO, "size": 7.3, "color": VIOLET})]],
            ["论文", "PipeDream(SOSP 2019)—— 1F1B 调度与异步流水",
             [("https://arxiv.org/abs/1806.03377", {"link": "https://arxiv.org/abs/1806.03377",
                                                   "font": MONO, "size": 7.3, "color": VIOLET})]],
            ["论文", "Megatron-LM(2019)—— TP 与 PP 的组合配方",
             [("https://arxiv.org/abs/1909.08053", {"link": "https://arxiv.org/abs/1909.08053",
                                                   "font": MONO, "size": 7.3, "color": VIOLET})]]]
    table(s, ML, 1.86, CW, data, [1.02, 5.42, 5.65], row_h=0.56, header_h=0.34, size=8.2,
          header_size=8.4)
    txt(s, ML, 6.28, CW, 0.24,
        "链接指向 vLLM 官方 latest 文档与 arXiv 原文;若文档路径调整,可在 docs.vllm.ai 站内按页面标题检索。",
        size=8.2, color=MUTED)
    footer(s, 18)
    notes(s, "参考页。三篇论文对应 P4 的三个里程碑;vLLM 文档对应 P15 的选型建议与 P16 的排错;"
             "源码仓库对应 P17 的索引表。")


# ----------------------------------------------------------------- slide 19 (Thanks)
def s19():
    s = prs.slides.add_slide(BLANK)
    shape(s, 0, 0, SW, SH, fill=INK, kind=MSO_SHAPE.RECTANGLE).line.fill.background()

    # right: four quiet bars, echoing the pipeline
    bar_cols = [RGBColor(0x24, 0x33, 0x4E), RGBColor(0x1F, 0x2C, 0x44),
                RGBColor(0x1B, 0x27, 0x3C), RGBColor(0x18, 0x23, 0x36)]
    for i in range(4):
        shape(s, 9.30, 1.62 + i * 0.98, 2.55 - i * 0.16, 0.66, fill=bar_cols[i],
              kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)

    rule = shape(s, ML, 2.28, 1.30, 0.05, fill=ORANGE, kind=MSO_SHAPE.RECTANGLE)
    rule.line.fill.background()
    txt(s, ML, 2.58, 7.6, 1.15, "Thanks", size=66, color=WHITE, bold=True)
    txt(s, ML, 3.86, 7.6, 0.36, "Q & A · 讨论与指正", size=18, color=RGBColor(0xC7, 0xD2, 0xE0))

    txt(s, ML, 6.48, 9.0, 0.26,
        "源码基准:vllm main @ 70c0016 · 仅 NVIDIA GPU(CUDA / NCCL)路径",
        size=9, color=RGBColor(0x7A, 0x8A, 0xA3))
    notes(s, "结束页。留出时间回答问题;如需深入某一块,可回到 P8–P14 的实现部分。")


# ----------------------------------------------------------------- build
def main():
    s01(); s02(); s03(); s04(); s05(); s06(); s07(); s08(); s09()
    s10(); s11(); s12(); s13(); s14(); s15(); s16(); s17(); s18(); s19()
    out = "assets/vllm-novidia/vllm-PP流水线并行-原理与部署.pptx"
    prs.save(out)
    print("saved:", out, "slides:", len(prs.slides))


if __name__ == "__main__":
    main()
