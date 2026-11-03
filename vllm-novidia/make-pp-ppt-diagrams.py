#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate PNG diagrams for the Pipeline-Parallelism (PP) PPT.

Run with the bundled Python (Pillow required):
    python make-pp-ppt-diagrams.py
Output: ppt-img/*.png next to this script.

All facts are drawn from the vLLM main branch (vllm repo checkout at main,
commit 70c0016). Pure Pillow drawing, no external deps.
"""

from __future__ import annotations

import math
import os
import sys

from PIL import Image, ImageDraw, ImageFont

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "ppt-img")
os.makedirs(OUT_DIR, exist_ok=True)

FONT_REG = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"

# palette (fill, stroke) pairs following the repo's existing diagram style
C = {
    "blue":   ("#E8F1FB", "#2B6CB0"),
    "green":  ("#EAF7EE", "#2F855A"),
    "orange": ("#FFF4E6", "#C05621"),
    "red":    ("#FDECEC", "#C53030"),
    "purple": ("#F0EBFF", "#553C9A"),
    "teal":   ("#E6FFFA", "#2C7A7B"),
    "pink":   ("#FFF0F6", "#B83280"),
    "gray":   ("#F2F4F7", "#4A5568"),
    "yellow": ("#FFFBEA", "#B7791F"),
}
TEXT_DARK = "#1A202C"
TEXT_SOFT = "#4A5568"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)


def text_w(d: ImageDraw.ImageDraw, s: str, f: ImageFont.FreeTypeFont) -> float:
    return d.textlength(s, font=f)


def rr(d, xy, fill, outline, radius=12, width=3):
    d.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def ctext(d, cx, cy, s, f, fill=TEXT_DARK):
    """single-line text centered at (cx, cy)"""
    w = text_w(d, s, f)
    asc, desc = f.getmetrics()
    h = asc + desc
    d.text((cx - w / 2, cy - h / 2), s, font=f, fill=fill)


def mtext(d, x, y, lines, f, fill=TEXT_DARK, lh=None, align="left", bold_first=False):
    """multi-line text block; returns bottom y"""
    if lh is None:
        lh = int(f.size * 1.45)
    for i, s in enumerate(lines):
        ff = f
        col = fill
        if bold_first and i == 0:
            ff = font(f.size, bold=True)
        if align == "center":
            w = text_w(d, s, ff)
            d.text((x - w / 2, y + i * lh), s, font=ff, fill=col)
        else:
            d.text((x, y + i * lh), s, font=ff, fill=col)
    return y + len(lines) * lh


def arrow(d, p0, p1, color="#2D3748", width=4, head=14, dashed=False):
    x0, y0 = p0
    x1, y1 = p1
    if dashed:
        # manually draw dashed line
        n = 26
        for i in range(n):
            if i % 2 == 0:
                t0 = i / n
                t1 = min((i + 0.55) / n, 1.0)
                d.line(
                    (x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0,
                     x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1),
                    fill=color, width=width)
    else:
        d.line((x0, y0, x1, y1), fill=color, width=width)
    # arrowhead
    ang = math.atan2(y1 - y0, x1 - x0)
    for da in (math.radians(152), math.radians(-152)):
        d.polygon(
            [(x1, y1),
             (x1 + head * math.cos(ang + da), y1 + head * math.sin(ang + da)),
             (x1 + head * math.cos(ang - da), y1 + head * math.sin(ang - da))],
            fill=color)


def caption(d, cx, y, s, f=None):
    f = f or font(26)
    w = text_w(d, s, f)
    d.rounded_rectangle((cx - w / 2 - 26, y - 12, cx + w / 2 + 26, y + 40),
                        radius=18, fill="#FFFFFF", outline="#CBD5E0", width=2)
    d.text((cx - w / 2, y), s, font=f, fill=TEXT_SOFT)


def save(img: Image.Image, name: str):
    path = os.path.join(OUT_DIR, name)
    img.save(path)
    print("saved", path)


W = 1840  # standard width


# ----------------------------------------------------------------------------
# 1. concept: what is PP
# ----------------------------------------------------------------------------
def diagram_concept():
    img = Image.new("RGB", (W, 980), "#FFFFFF")
    d = ImageDraw.Draw(img)
    fT = font(40, True)
    fB = font(28)
    fS = font(24)

    ctext(d, W / 2, 60, "流水线并行(Pipeline Parallelism):按“层”把模型切开,每张 GPU 只负责一段", fT)

    # --- left: full model tower -------------------------------------------
    bx, bw = 130, 300
    y = 150
    layers = [
        ("LM Head(输出词表)", "pink"),
        ("Norm(归一化)", "pink"),
        ("Transformer 层 L24 ~ L31", "teal"),
        ("Transformer 层 L16 ~ L23", "teal"),
        ("Transformer 层 L8 ~ L15", "teal"),
        ("Transformer 层 L0 ~ L7", "teal"),
        ("Embedding(词嵌入)", "blue"),
    ]
    lh = 82
    d.rounded_rectangle((bx - 24, y - 24, bx + bw + 24, y + len(layers) * lh + 24),
                        radius=16, outline="#A0AEC0", width=3)
    ctext(d, bx + bw / 2, y - 44, "完整模型(例如 32 层 Llama)", font(28, True), TEXT_SOFT)
    for i, (name, kind) in enumerate(layers):
        fill, stroke = C[kind]
        yy = y + i * lh
        rr(d, (bx, yy + 6, bx + bw, yy + lh - 6), fill, stroke, 10, 2)
        ctext(d, bx + bw / 2, yy + lh / 2, name, font(23, True))
    # dashed cut lines between segments
    for k in (1, 3, 5):
        yy = y + k * lh
        for i in range(14):
            d.line((bx - 34 + i * 26, yy, bx - 8 + i * 26, yy), fill="#E53E3E", width=4)
    ctext(d, bx + bw / 2, y + len(layers) * lh + 48, "红色虚线 = 切分点(4 段)", fS, "#C53030")

    # --- right: 4 GPUs in a pipeline row ----------------------------------
    gx0, gw = 580, 215
    gy = 220
    gap = 75
    names = ["GPU 0(首阶段)", "GPU 1", "GPU 2", "GPU 3(末阶段)"]
    segs = ["Embedding\nL0 ~ L7", "L8 ~ L15", "L16 ~ L23", "L24 ~ L31\nNorm + LM Head"]
    kinds = ["blue", "green", "orange", "pink"]
    for i in range(4):
        x = gx0 + i * (gw + gap)
        fill, stroke = C["gray"]
        # GPU card
        rr(d, (x, gy, x + gw, gy + 220), fill, stroke, 14, 3)
        ctext(d, x + gw / 2, gy + 40, names[i], font(24, True))
        fill2, stroke2 = C[kinds[i]]
        rr(d, (x + 22, gy + 74, x + gw - 22, gy + 186), fill2, stroke2, 10, 2)
        for j, line in enumerate(segs[i].split("\n")):
            ctext(d, x + gw / 2, gy + 112 + j * 38, line, font(21))
        # arrows between GPUs
        if i < 3:
            ax = x + gw
            ay = gy + 130
            arrow(d, (ax + 10, ay), (ax + gap - 10, ay), "#C05621", 5, 14)
            ctext(d, ax + gap / 2, ay - 30, "隐状态", font(19), "#C05621")
    # input / output
    ctext(d, gx0 - 100, gy + 64, "输入 token", font(21), "#2B6CB0")
    arrow(d, (gx0 - 130, gy + 130), (gx0 - 16, gy + 130), "#2B6CB0", 5, 15)
    xe = gx0 + 3 * (gw + gap) + gw
    arrow(d, (xe + 14, gy + 130), (xe + 100, gy + 130), "#B83280", 5, 15)
    ctext(d, xe + 57, gy + 60, "logits", font(20), "#B83280")
    ctext(d, xe + 57, gy + 92, "采样得 token", font(19), "#B83280")

    # caption
    caption(d, W / 2, 790, "每张 GPU 显存里只有自己那 8 层的权重 → 权重显存降为 1/4,KV Cache 显存也降为 1/4")
    caption(d, W / 2, 866, "代价:一个请求的推理必须按顺序流过 4 张卡 → 延迟变高,需要“流水线”填充来保吞吐")

    save(img, "pp-01-concept.png")


# ----------------------------------------------------------------------------
# 2. forward dataflow with communication
# ----------------------------------------------------------------------------
def diagram_dataflow():
    img = Image.new("RGB", (W, 940), "#FFFFFF")
    d = ImageDraw.Draw(img)
    fT = font(40, True)
    fS = font(23)
    fSS = font(20)

    ctext(d, W / 2, 56, "一次前向传播:数据在各阶段之间“接力”(以 PP=4、每段 8 层为例)", fT)

    # input
    inx = 80
    rr(d, (inx, 250, inx + 170, 420), *C["blue"], 12, 3)
    ctext(d, inx + 85, 300, "输入", font(26, True))
    mtext(d, inx + 85, 328, ["input_ids", "positions"], font(21), align="center")

    # four stages
    sx = 310
    sw, sh = 215, 260
    gap = 145
    stage_names = ["阶段 0(首)", "阶段 1", "阶段 2", "阶段 3(末)"]
    titles = [
        ["Embedding", "L0 ~ L7"],
        ["L8 ~ L15"],
        ["L16 ~ L23"],
        ["L24 ~ L31", "Norm+LMHead"],
    ]
    extras = [
        ["多模态 encoder", "也只在首阶段"],
        ["只持有这 8 层的", "权重和 KV Cache"],
        ["只持有这 8 层的", "权重和 KV Cache"],
        ["采样也在这里", "(logits → token)"],
    ]
    kinds = ["blue", "green", "orange", "pink"]
    for i in range(4):
        x = sx + i * (sw + gap)
        fill, stroke = C[kinds[i]]
        rr(d, (x, 180, x + sw, 180 + sh), fill, stroke, 14, 3)
        ctext(d, x + sw / 2, 214, stage_names[i], font(24, True), stroke)
        yy = 246
        for line in titles[i]:
            rr(d, (x + 18, yy, x + sw - 18, yy + 50), "#FFFFFF", stroke, 8, 2)
            ctext(d, x + sw / 2, yy + 25, line, font(21, True))
            yy += 60
        mtext(d, x + sw / 2, yy + 10, extras[i], fSS, TEXT_SOFT, lh=30, align="center")
        # arrows between stages
        if i < 3:
            ax = x + sw
            ay = 310
            arrow(d, (ax + 10, ay), (ax + gap - 10, ay), "#C05621", 5, 14)
            ctext(d, ax + gap / 2, ay - 30, "hidden_states", font(17), "#C05621")
            ctext(d, ax + gap / 2, ay + 30, "+ residual", font(17), "#C05621")

    # arrow input->stage0
    arrow(d, (inx + 178, 335), (sx - 14, 335), "#2B6CB0", 5, 15)

    # output
    ox = sx + 3 * (sw + gap) + sw
    rr(d, (ox + 20, 250, ox + 210, 420), *C["pink"], 12, 3)
    ctext(d, ox + 115, 300, "输出", font(26, True))
    mtext(d, ox + 115, 328, ["logits", "采样 next token"], font(21), align="center")
    arrow(d, (ox + 2, 335), (ox + 14, 335), "#B83280", 5, 15)

    # backward broadcast (dashed, from stage-3 bottom)
    bx = ox + sw // 2
    arrow(d, (bx, 180 + sh + 6), (bx, 552), "#553C9A", 4, 13, dashed=True)
    for j, line in enumerate(["末阶段采样结果 broadcast 回传", "(各阶段用它同步本地状态)"]):
        w = text_w(d, line, fSS)
        d.text((bx - 24 - w, 466 + j * 30), line, font=fSS, fill="#553C9A")

    # communication band
    y0 = 560
    d.rounded_rectangle((100, y0, W - 100, y0 + 250), radius=16,
                        fill="#FFFBF5", outline="#C05621", width=3)
    ctext(d, W / 2, y0 + 44, "阶段间怎么传?(vLLM main 真实实现)", font(27, True), "#C05621")
    mtext(d, W / 2, y0 + 88, [
        "① isend / irecv(NCCL 点对点):上一阶段把 {hidden_states, residual} 异步发给 rank+1,下一阶段异步收",
        "② 元数据走 CPU(gloo):先 send_object 通知“有几个张量、什么形状”,再收张量本体(GPU/NCCL)",
        "③ 非阻塞:发送方把 handle 存起来,下一轮开头才 wait;接收方包成 AsyncIntermediateTensors 懒同步",
        "④ TP>1 时的优化:每个张量只发本 TP rank 的切片,接收方在 TP 组内 all-gather 还原 → 通信量降为 1/TP",
    ], fS, TEXT_DARK, lh=40, align="center")

    save(img, "pp-02-dataflow.png")


# ----------------------------------------------------------------------------
# 3. pipeline bubble vs filled pipeline
# ----------------------------------------------------------------------------
def diagram_bubble():
    img = Image.new("RGB", (W, 1010), "#FFFFFF")
    d = ImageDraw.Draw(img)
    fT = font(38, True)
    fS = font(23)
    fSS = font(20)

    ctext(d, W / 2, 54, "流水线的“气泡”:多批次错开执行,让所有 GPU 都忙起来", fT)

    batch_colors = {
        "A": ("#2B6CB0", "#E8F1FB"),
        "B": ("#2F855A", "#EAF7EE"),
        "C": ("#C05621", "#FFF4E6"),
        "D": ("#553C9A", "#F0EBFF"),
    }

    def gantt(y0, title, schedule, note):
        """schedule: dict[(gpu,slot)] = batch letter or None"""
        d.text((120, y0 - 62), title, font=font(27, True), fill=TEXT_DARK)
        gx0, cellw, cellh = 340, 130, 62
        # gpu labels
        for g in range(4):
            ctext(d, gx0 - 80, y0 + g * cellh + cellh / 2, f"GPU{g}", font(23, True), TEXT_SOFT)
        for s in range(8):
            x = gx0 + s * cellw
            d.text((x + 8, y0 - 26), f"t{s + 1}", font=fSS, fill=TEXT_SOFT)
        # cells
        for g in range(4):
            for s in range(8):
                x = gx0 + s * cellw
                y = y0 + g * cellh
                letter = schedule.get((g, s))
                if letter:
                    stroke, fill = batch_colors[letter]
                    rr(d, (x + 5, y + 6, x + cellw - 5, y + cellh - 6), fill, stroke, 8, 2)
                    ctext(d, x + cellw / 2, y + cellh / 2, letter, font(24, True), stroke)
                else:
                    # bubble cell
                    d.rectangle((x + 5, y + 6, x + cellw - 5, y + cellh - 6),
                                fill="#F7FAFC", outline="#E2E8F0", width=1)
                    for k in range(6):
                        d.line((x + 10 + k * 22, y + 10, x + 24 + k * 22, y + cellh - 10),
                               fill="#FEB2B2", width=2)
        ctext(d, W / 2, y0 + 4 * cellh + 26, note, fS, TEXT_SOFT)
        return y0 + 4 * cellh + 60

    # top: serial
    sched1 = {}
    batches = ["A", "B", "C", "D"]
    for bi, b in enumerate(batches):
        for g in range(4):
            sched1[(g, bi * 4 + g)] = b
    y = gantt(175, "× 同一时刻只有 1 个批次在流水线里", sched1,
              "每格深红斜纹 = GPU 空转(“气泡”)。串行执行时大部分时间都在等别的阶段算完")

    # bottom: filled
    sched2 = {}
    for bi, b in enumerate(batches):
        for g in range(4):
            sched2[(g, bi + g)] = b
    y = gantt(y + 60, "√ batch queue 维持 pp_size=4 个批次同时在飞(vLLM main 的做法)", sched2,
              "批次 A/B/C/D 依次错开 1 拍进入 → 只有启动和排空的几格是气泡,稳态时 4 张卡全忙")

    save(img, "pp-03-bubble.png")


# ----------------------------------------------------------------------------
# 4. rank layout / process groups
# ----------------------------------------------------------------------------
def diagram_ranks():
    img = Image.new("RGB", (W, 950), "#FFFFFF")
    d = ImageDraw.Draw(img)
    fT = font(38, True)
    fS = font(23)
    fSS = font(20)

    ctext(d, W / 2, 56, "8 张 GPU,TP=2 + PP=4:进程组是怎么划分的?", fT)
    ctext(d, W / 2, 108, "parallel_state.initialize_model_parallel():rank 布局为 [DP × PP × PCP × TP] 五维数组", fS, TEXT_SOFT)

    # 8 gpu squares
    n = 8
    gw = 150
    gap = (W - 240 - n * gw) / (n - 1)
    y0 = 290
    for i in range(n):
        x = 120 + i * (gw + gap)
        fill, stroke = C["teal"]
        rr(d, (x, y0, x + gw, y0 + 130), fill, stroke, 12, 3)
        ctext(d, x + gw / 2, y0 + 44, f"rank {i}", font(25, True))
        ctext(d, x + gw / 2, y0 + 86, f"GPU {i}", font(23), TEXT_SOFT)

    # PP group bars (orange) above cards, two stacked rows
    pp_groups = [(0, 2, 4, 6), (1, 3, 5, 7)]
    for k, grp in enumerate(pp_groups):
        ypp = y0 - 136 + k * 66
        x0 = 120 + grp[0] * (gw + gap)
        x1 = 120 + grp[3] * (gw + gap) + gw
        d.rounded_rectangle((x0 - 8, ypp, x1 + 8, ypp + 56), radius=12, fill=C["orange"][0], outline=C["orange"][1], width=3)
        ctext(d, (x0 + x1) / 2, ypp + 28, f"PP 组 {k}:[rank {','.join(map(str, grp))}] = 阶段 0→1→2→3", font(22, True), C["orange"][1])

    # TP group bars (blue) under cards
    ytp = y0 + 160
    for k in range(4):
        x0 = 120 + (2 * k) * (gw + gap)
        x1 = x0 + 2 * gw + gap
        d.rounded_rectangle((x0 - 8, ytp, x1 + 8, ytp + 54), radius=12, fill=C["blue"][0], outline=C["blue"][1], width=3)
        ctext(d, (x0 + x1) / 2, ytp + 27, f"TP 组 {k}", font(22, True), C["blue"][1])
    d.text((120, ytp + 72), "TP 组:组内 2 张卡同算一层(all-reduce 合并结果),共 4 组", font=fSS, fill=C["blue"][1])

    # code note
    ycode = ytp + 130
    d.rounded_rectangle((120, ycode, W - 120, ycode + 210), radius=14, fill="#1A202C")
    code = [
        "all_ranks = arange(8).reshape(DP=1, PP=4, PCP=1, TP=2)          # 布局顺序",
        "PP 组  = transpose(PP维, TP维) 后按 PP 维切开 → [0,2,4,6] / [1,3,5,7]",
        "TP 组  = 按 TP 维直接切开     → [0,1] [2,3] [4,5] [6,7]",
        "PP 组内:rank+1 是我的下一阶段(isend 目标),rank-1 是上一阶段(irecv 来源)",
    ]
    fcode = font(23)
    for i, line in enumerate(code):
        d.text((160, ycode + 22 + i * 42), line, font=fcode, fill="#CBD5E0")

    caption(d, W / 2, ycode + 246, "同一张卡既属于一个 TP 组,也属于一个 PP 组 —— TP 与 PP 是正交的,可以任意组合")

    save(img, "pp-04-ranks.png")


# ----------------------------------------------------------------------------
# 5. architecture: where PP lives in vLLM V1
# ----------------------------------------------------------------------------
def diagram_arch():
    img = Image.new("RGB", (W, 880), "#FFFFFF")
    d = ImageDraw.Draw(img)
    fT = font(38, True)
    fS = font(22)
    fSS = font(19)

    ctext(d, W / 2, 52, "vLLM V1 引擎中的 PP:EngineCore 调度一次,广播给所有阶段并行执行", fT)

    # EngineCore box (left)
    ex, ey, ew, eh = 90, 170, 320, 400
    rr(d, (ex, ey, ex + ew, ey + eh), *C["red"], 16, 3)
    ctext(d, ex + ew / 2, ey + 40, "EngineCore 进程", font(25, True), C["red"][1])
    rr(d, (ex + 22, ey + 74, ex + ew - 22, ey + 178), "#FFFFFF", C["red"][1], 10, 2)
    ctext(d, ex + ew / 2, ey + 106, "Scheduler 调度器", font(21, True))
    ctext(d, ex + ew / 2, ey + 140, "算出每步跑哪些请求", font(18), TEXT_SOFT)
    rr(d, (ex + 22, ey + 198, ex + ew - 22, ey + 302), "#FFFFFF", C["red"][1], 10, 2)
    ctext(d, ex + ew / 2, ey + 230, "batch queue", font(21, True))
    ctext(d, ex + ew / 2, ey + 264, "深度 = pp_size(4)", font(18), TEXT_SOFT)
    rr(d, (ex + 22, ey + 322, ex + ew - 22, ey + eh - 16), "#FFFFFF", C["red"][1], 10, 2)
    ctext(d, ex + ew / 2, ey + 352, "用采样结果更新请求", font(18), TEXT_SOFT)
    ctext(d, ex + ew / 2, ey + 380, "状态(供下一轮调度)", font(18), TEXT_SOFT)

    # Executor box (middle)
    mx, my, mw, mh = 480, 290, 230, 180
    rr(d, (mx, my, mx + mw, my + mh), *C["yellow"], 14, 3)
    ctext(d, mx + mw / 2, my + 40, "Executor 执行器", font(23, True), C["yellow"][1])
    mtext(d, mx + mw / 2, my + 84, ["单机:mp(多进程)", "多机:ray", "都支持 PP"], fS, align="center", lh=34)

    # EngineCore -> Executor
    arrow(d, (ex + ew + 8, my + 90), (mx - 12, my + 90), "#C53030", 4, 13)
    ctext(d, (ex + ew + mx) / 2, my + 52, "提交批次", font(16), "#C53030")

    # workers (right)
    wx0, ww, wh = 770, 205, 210
    gap = (W - 90 - wx0 - 4 * ww) / 3
    wy = 200
    kinds = ["blue", "green", "orange", "pink"]
    wnames = ["Worker 0", "Worker 1", "Worker 2", "Worker 3"]
    stages = [["阶段 0", "Embed + L0~7"], ["阶段 1", "L8~15"],
              ["阶段 2", "L16~23"], ["阶段 3", "Norm + LMHead"]]
    for i in range(4):
        x = wx0 + i * (ww + gap)
        fill, stroke = C[kinds[i]]
        rr(d, (x, wy, x + ww, wy + wh), fill, stroke, 14, 3)
        ctext(d, x + ww / 2, wy + 34, wnames[i], font(22, True), stroke)
        ctext(d, x + ww / 2, wy + 72, stages[i][0], font(19, True), TEXT_DARK)
        ctext(d, x + ww / 2, wy + 100, stages[i][1], font(18), TEXT_DARK)
        rr(d, (x + 14, wy + 124, x + ww - 14, wy + 204), "#FFFFFF", stroke, 8, 2)
        ctext(d, x + ww / 2, wy + 140, "ModelRunner", font(16, True), TEXT_SOFT)
        ctext(d, x + ww / 2, wy + 162, "自己的 KV Cache", font(13), TEXT_SOFT)
        ctext(d, x + ww / 2, wy + 184, "+ CUDA Graph", font(13), TEXT_SOFT)
        # P2P arrows between adjacent workers
        if i < 3:
            ax = x + ww
            ay = wy + 96
            arrow(d, (ax + 6, ay), (ax + gap - 6, ay), "#C05621", 4, 12)

    # P2P note under workers
    ctext(d, wx0 + (4 * ww + 3 * gap) / 2, wy + wh + 30,
          "相邻阶段:isend / irecv 隐状态(NCCL 点对点,非阻塞)", font(19), "#C05621")

    # broadcast bus: Executor right -> up -> horizontal bus -> down to each worker
    busy = 130
    d.line((mx + mw, my + 60, wx0 - 40, my + 60), fill="#B7791F", width=4)
    d.line((wx0 - 40, my + 60, wx0 - 40, busy), fill="#B7791F", width=4)
    last_cx = wx0 + 3 * (ww + gap) + ww / 2
    d.line((wx0 - 40, busy, last_cx, busy), fill="#B7791F", width=4)
    for i in range(4):
        cx = wx0 + i * (ww + gap) + ww / 2
        arrow(d, (cx, busy + 4), (cx, wy - 10), "#B7791F", 3, 11)
    d.text((wx0 - 30, busy - 34), "SchedulerOutput 广播:每个阶段拿到同一份调度结果", font=font(19), fill="#B7791F")

    # output path back (purple band)
    ybot = wy + wh + 80
    arrow(d, (last_cx, wy + wh + 44), (last_cx, ybot - 10), "#553C9A", 4, 13)
    d.rounded_rectangle((wx0, ybot, W - 90, ybot + 120), radius=14, fill=C["purple"][0], outline=C["purple"][1], width=3)
    ctext(d, (wx0 + W - 90) / 2, ybot + 34, "末阶段(阶段 3)采样 → pp_broadcast 采样结果回传所有阶段", font(21, True), C["purple"][1])
    ctext(d, (wx0 + W - 90) / 2, ybot + 74, "ModelRunnerOutput 只从 output_rank(末阶段的 TP0)返回给 EngineCore", font(18), TEXT_SOFT)

    # key files
    yfiles = ybot + 160
    d.rounded_rectangle((90, yfiles, W - 90, yfiles + 96), radius=14, fill="#F7FAFC", outline="#A0AEC0", width=2)
    ctext(d, W / 2, yfiles + 30, "关键源码(vllm 仓 main 分支):", font(20, True), TEXT_SOFT)
    ctext(d, W / 2, yfiles + 64,
          "v1/engine/core.py(batch queue)· v1/executor/multiproc_executor.py(广播)· v1/worker/gpu_worker.py(irecv/isend)",
          font(19), TEXT_DARK)

    save(img, "pp-05-arch.png")


# ----------------------------------------------------------------------------
# 6. worker execute_model four steps
# ----------------------------------------------------------------------------
def diagram_exec():
    img = Image.new("RGB", (W, 810), "#FFFFFF")
    d = ImageDraw.Draw(img)
    fT = font(38, True)
    fS = font(22)
    fSS = font(19)

    ctext(d, W / 2, 54, "每个 Worker 每步做的事(gpu_worker.execute_model 核心四步)", fT)

    steps = [
        ("① 等上轮发送", "wait 上一轮 isend\n的 handle 全部落账", "blue"),
        ("② 异步收隐状态", "非首阶段:irecv_tensor_dict\n元数据(gloo)+张量(NCCL)", "green"),
        ("③ 本阶段前向", "跑自己那 8 层\n读写自己的 KV Cache", "orange"),
        ("④ 异步发下一站", "非末阶段:isend_tensor_dict\nhandle 存着,下轮再等", "pink"),
    ]
    bw, bh = 350, 250
    gap = (W - 200 - 4 * bw) / 3
    y0 = 150
    for i, (title, body, kind) in enumerate(steps):
        x = 100 + i * (bw + gap)
        fill, stroke = C[kind]
        rr(d, (x, y0, x + bw, y0 + bh), fill, stroke, 14, 3)
        ctext(d, x + bw / 2, y0 + 42, title, font(24, True), stroke)
        for j, line in enumerate(body.split("\n")):
            ctext(d, x + bw / 2, y0 + 96 + j * 36, line, font(19), TEXT_DARK)
        if i < 3:
            arrow(d, (x + bw + 8, y0 + bh / 2), (x + bw + gap - 8, y0 + bh / 2), "#4A5568", 4, 13)

    # then: sampling split
    y1 = 470
    rr(d, (100, y1, W / 2 - 40, y1 + 170), *C["purple"], 14, 3)
    ctext(d, (100 + W / 2 - 40) / 2, y1 + 40, "末阶段:execute_model 直接返回", font(23, True), C["purple"][1])
    mtext(d, (100 + W / 2 - 40) / 2, y1 + 84, [
        "hidden_states → compute_logits → 采样",
        "pp_broadcast(采样 token)发给所有阶段",
    ], fS, align="center", lh=36)
    rr(d, (W / 2 + 40, y1, W - 100, y1 + 170), *C["teal"], 14, 3)
    ctext(d, (W / 2 + 40 + W - 100) / 2, y1 + 40, "非末阶段:execute_model 返回 None", font(23, True), C["teal"][1])
    mtext(d, (W / 2 + 40 + W - 100) / 2, y1 + 84, [
        "但还没完 —— sample_tokens 阶段",
        "pp_receive 收采样结果,postprocess 同步本地状态",
    ], fS, align="center", lh=36)

    caption(d, W / 2, y1 + 220,
            "②④ 都是非阻塞的:隐状态的收发与本阶段的计算可以重叠,这正是流水线能“填满”的原因")

    save(img, "pp-06-exec.png")


# ----------------------------------------------------------------------------
# 7. layer split algorithm
# ----------------------------------------------------------------------------
def diagram_split():
    img = Image.new("RGB", (W, 930), "#FFFFFF")
    d = ImageDraw.Draw(img)
    fT = font(38, True)
    fS = font(23)
    fSS = font(20)

    ctext(d, W / 2, 54, "层怎么切?get_pp_indices():尽量均衡,余数照顾中间段", fT)
    ctext(d, W / 2, 106, "例:33 层模型切成 PP=4 → 33 ÷ 4 = 8 余 1 → 分段 [8, 8, 9, 8](余数给中间段)", fS, TEXT_SOFT)

    # 33 layer strips stacked, 4 segments
    x0, x1 = 360, 660
    y0 = 160
    lh = 15
    total = 33
    parts = [(0, 8, "blue", "阶段 0\n8 层"), (8, 8, "green", "阶段 1\n8 层"),
             (16, 9, "orange", "阶段 2\n9 层(余数)"), (25, 8, "pink", "阶段 3\n8 层")]
    yy = y0
    for start, cnt, kind, label in parts:
        fill, stroke = C[kind]
        seg_h = cnt * lh
        d.rounded_rectangle((x0, yy, x1, yy + seg_h), radius=6, fill=fill, outline=stroke, width=2)
        ctext(d, (x0 + x1) / 2, yy + seg_h / 2, f"L{start} ~ L{start + cnt - 1}({cnt} 层)", font(19, True))
        # side label
        d.rounded_rectangle((x1 + 40, yy + seg_h / 2 - 40, x1 + 220, yy + seg_h / 2 + 40),
                            radius=10, fill=fill, outline=stroke, width=2)
        for j, line in enumerate(label.split("\n")):
            ctext(d, x1 + 130, yy + seg_h / 2 - 12 + j * 26, line, font(19))
        yy += seg_h + 6
    # embed / lmhead annotations
    d.rounded_rectangle((x0 - 245, y0 - 66, x0 - 20, y0 - 8), radius=10,
                        fill=C["gray"][0], outline=C["gray"][1], width=2)
    ctext(d, x0 - 132, y0 - 37, "Embedding 在首段", font(19, True))
    yend = yy
    d.rounded_rectangle((x0 - 245, yend + 8, x0 - 20, yend + 66), radius=10,
                        fill=C["gray"][0], outline=C["gray"][1], width=2)
    ctext(d, x0 - 132, yend + 37, "Norm+LMHead 在末段", font(19, True))

    # right notes
    nx = 1000
    d.rounded_rectangle((nx, 170, W - 110, 830), radius=16, fill="#FFFBF5", outline="#C05621", width=3)
    ctext(d, (nx + W - 110) / 2, 210, "为什么余数给中间段?", font(25, True), "#C05621")
    mtext(d, nx + 40, 258, [
        "• 首段:多一个 Embedding(词表很大)",
        "• 末段:多一个 Norm + LM Head(词表很大)",
        "• 首末本身计算就更重,再加分层就不均衡了",
        "",
        "vLLM 会打印日志:",
        "  Hidden layers were unevenly partitioned:",
        "  [8,8,9,8]  ← 不满足可手动指定",
        "",
        "环境变量(手动精确控制):",
        "  VLLM_PP_LAYER_PARTITION=\"8,8,9,8\"",
        "  要求:段数 = pp_size,总和 = 总层数",
        "",
        "每个阶段的层范围:",
        "  start = 前面所有段之和",
        "  end   = start + 本段层数",
        "  (model_config.get_layers_start_end_indices)",
    ], fS, TEXT_DARK, lh=38)

    save(img, "pp-07-split.png")


# ----------------------------------------------------------------------------
# 8. TP vs PP comparison
# ----------------------------------------------------------------------------
def diagram_compare():
    img = Image.new("RGB", (W, 990), "#FFFFFF")
    d = ImageDraw.Draw(img)
    fT = font(38, True)
    fS = font(22)
    fSS = font(19)

    ctext(d, W / 2, 54, "TP 与 PP 的分工:什么时候用哪个?", fT)

    # left panel: TP
    lx, ly, lw, lh = 100, 120, 780, 610
    d.rounded_rectangle((lx, ly, lx + lw, ly + lh), radius=16, fill="#F7FAFF", outline=C["blue"][1], width=3)
    ctext(d, lx + lw / 2, ly + 44, "TP 张量并行:切“层内”的矩阵", font(27, True), C["blue"][1])
    # two GPU columns with 3 layers each, comm arrows each layer
    gw, gh, gapx = 200, 76, 240
    gy = ly + 105
    for col in range(2):
        x = lx + 120 + col * (gw + gapx)
        for row in range(3):
            y = gy + row * (gh + 32)
            rr(d, (x, y, x + gw, y + gh), *C["blue"], 10, 2)
            ctext(d, x + gw / 2, y + gh / 2, f"层{row}: 左半边" if col == 0 else f"层{row}: 右半边", font(19, True))
        ctext(d, x + gw / 2, gy + 3 * (gh + 32) + 10, f"GPU {col}", font(21, True), TEXT_SOFT)
    for row in range(3):
        y = gy + row * (gh + 32) + gh / 2
        x0 = lx + 120 + gw
        x1 = x0 + gapx
        # double arrow (all-reduce)
        arrow(d, (x0 + 8, y - 12), (x1 - 8, y - 12), "#C53030", 3, 10)
        arrow(d, (x1 - 8, y + 14), (x0 + 8, y + 14), "#C53030", 3, 10)
    ctext(d, lx + 120 + gw + gapx / 2, gy - 18, "每一层都要 all-reduce", fSS, "#C53030")
    mtext(d, lx + lw / 2, gy + 3 * (gh + 32) + 56, [
        "• 通信极频繁、量极大 → 需要 NVLink 等高带宽",
        "• 适合节点内;单请求延迟低(一层多人同时算)",
        "• 解决“单层/整卡放不下”的问题",
    ], fS, align="center", lh=38)

    # right panel: PP
    rx = 960
    d.rounded_rectangle((rx, ly, rx + lw, ly + lh), radius=16, fill="#FFF9F4", outline=C["orange"][1], width=3)
    ctext(d, rx + lw / 2, ly + 44, "PP 流水线并行:切“层与层”", font(27, True), C["orange"][1])
    # 4 GPU boxes in a row
    gw2, gh2, gap2 = 150, 110, 66
    px = rx + 62
    py = gy + 90
    kinds = ["blue", "green", "orange", "pink"]
    labels = ["L0~7", "L8~15", "L16~23", "L24~31"]
    for i in range(4):
        x = px + i * (gw2 + gap2)
        rr(d, (x, py, x + gw2, py + gh2), *C[kinds[i]], 10, 2)
        ctext(d, x + gw2 / 2, py + gh2 / 2 - 12, f"GPU {i}", font(19, True))
        ctext(d, x + gw2 / 2, py + gh2 / 2 + 14, labels[i], font(17))
        if i < 3:
            arrow(d, (x + gw2 + 5, py + gh2 / 2), (x + gw2 + gap2 - 5, py + gh2 / 2), "#C05621", 4, 11)
    ctext(d, rx + lw / 2, py - 30, "只在阶段边界点对点传一次", fSS, "#C05621")
    mtext(d, rx + lw / 2, py + gh2 + 70, [
        "• 通信少、点对点 → 普通网络也行,可跨节点",
        "• 有流水线气泡、单请求延迟高",
        "• 解决“层数太多/跨节点放不下”的问题",
    ], fS, align="center", lh=38)

    # bottom guidance band
    yb = ly + lh + 26
    d.rounded_rectangle((100, yb, W - 100, yb + 208), radius=16, fill=C["green"][0], outline=C["green"][1], width=3)
    ctext(d, W / 2, yb + 40, "官方建议(docs/serving/parallelism_scaling.md)", font(24, True), C["green"][1])
    mtext(d, W / 2, yb + 88, [
        "• 单卡放不下、单机放得下 → 节点内 TP",
        "• 跨节点 → 节点内 TP × 节点间 PP(如 2 机×8 卡:--tp 8 --pp 2)",
        "• GPU 没有 NVLink(如 L40S)、或卡数与层数不整除 → 优先 PP(支持不均匀切分)",
    ], fS, align="center", lh=38)

    save(img, "pp-08-compare.png")


def main():
    diagram_concept()
    diagram_dataflow()
    diagram_bubble()
    diagram_ranks()
    diagram_arch()
    diagram_exec()
    diagram_split()
    diagram_compare()
    print("all done")


if __name__ == "__main__":
    main()
