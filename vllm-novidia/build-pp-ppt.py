#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the Pipeline-Parallelism (PP) PPT for vLLM main branch (GPU).

Run with the bundled Python (python-pptx + Pillow required):
    python build-pp-ppt.py
Output: vllm-main-PP流水线并行详解.pptx next to this script.

Content is based on the vllm repo main branch (commit 70c0016), GPU path only.
"""

from __future__ import annotations

import os
import sys

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "ppt-img")
OUT = os.path.join(HERE, "vllm-main-PP流水线并行详解.pptx")

# ---------------------------------------------------------------- palette
NAVY = RGBColor(0x1F, 0x3B, 0x66)      # title bar
BLUE = RGBColor(0x2B, 0x6C, 0xB0)
ORANGE = RGBColor(0xC0, 0x56, 0x21)
GREEN = RGBColor(0x2F, 0x85, 0x5A)
PURPLE = RGBColor(0x55, 0x3C, 0x9A)
RED = RGBColor(0xC5, 0x30, 0x30)
TEAL = RGBColor(0x2C, 0x7A, 0x7B)
DARK = RGBColor(0x1A, 0x20, 0x2C)
SOFT = RGBColor(0x4A, 0x55, 0x68)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_BG = RGBColor(0xF7, 0xFA, 0xFC)
CODE_BG = RGBColor(0x1A, 0x20, 0x2C)
CODE_FG = RGBColor(0xE2, 0xE8, 0xF0)
CODE_HL = RGBColor(0xF6, 0xAD, 0x55)

FONT = "微软雅黑"
CODE_FONT = "Consolas"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = 13.333, 7.5

page_no = 0


def new_slide():
    global page_no
    page_no += 1
    return prs.slides.add_slide(BLANK)


def rect(slide, x, y, w, h, fill, line=None, shadow=False):
    from pptx.enum.shapes import MSO_SHAPE
    sp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                                Inches(w), Inches(h))
    sp.fill.solid()
    sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line
        sp.line.width = Pt(1.2)
    sp.shadow.inherit = False
    return sp


def rrect(slide, x, y, w, h, fill, line=None):
    from pptx.enum.shapes import MSO_SHAPE
    sp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                                Inches(w), Inches(h))
    sp.adjustments[0] = 0.08
    sp.fill.solid()
    sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line
        sp.line.width = Pt(1.2)
    sp.shadow.inherit = False
    return sp


def text(slide, x, y, w, h, runs, size=18, color=DARK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, line_spacing=1.12, space_after=4, font_name=FONT):
    """runs: str 或 [(txt, {size,color,bold,font}), ...] 或 [[runs_para1],[runs_para2]]"""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    if isinstance(runs, str):
        runs = [[(runs, {})]]
    elif runs and isinstance(runs[0], tuple):
        runs = [runs]
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        p.space_after = Pt(space_after)
        for txt, opt in para:
            r = p.add_run()
            r.text = txt
            r.font.size = Pt(opt.get("size", size))
            r.font.bold = opt.get("bold", bold)
            r.font.color.rgb = opt.get("color", color)
            r.font.name = opt.get("font", font_name)
    return tb


def bullets(slide, x, y, w, h, items, size=17, gap=6, line_spacing=1.15):
    """items: list of (level, text) or (level, [(txt,opt),...])"""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    first = True
    for level, content in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.line_spacing = line_spacing
        p.space_after = Pt(gap)
        indent = "    " * level
        marker = "▪ " if level == 0 else "– "
        if level == 0:
            rr_ = p.add_run(); rr_.text = marker
            rr_.font.size = Pt(size); rr_.font.bold = True
            rr_.font.color.rgb = BLUE; rr_.font.name = FONT
        else:
            rr_ = p.add_run(); rr_.text = indent + marker
            rr_.font.size = Pt(size - 1); rr_.font.color.rgb = SOFT; rr_.font.name = FONT
        if isinstance(content, str):
            content = [(content, {})]
        for txt, opt in content:
            r = p.add_run()
            r.text = txt
            r.font.size = Pt(opt.get("size", size if level == 0 else size - 1))
            r.font.bold = opt.get("bold", False)
            r.font.color.rgb = opt.get("color", DARK if level == 0 else SOFT)
            r.font.name = opt.get("font", FONT)
    return tb


def code_block(slide, x, y, w, h, lines, size=13, title=None, highlight=()):
    """lines: list[str]; highlight: set of line indices to color differently"""
    pad = 0.10
    bh = h + (0.34 if title else 0)
    rect(slide, x, y, w, bh, CODE_BG)
    if title:
        text(slide, x + 0.16, y + 0.055, w - 0.3, 0.3,
             title, size=12, color=CODE_HL, bold=True, font_name=FONT)
    tb = slide.shapes.add_textbox(Inches(x + 0.16), Inches(y + (0.36 if title else pad)),
                                  Inches(w - 0.3), Inches(h - 2 * pad))
    tf = tb.text_frame
    tf.word_wrap = False
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = 1.08
        r = p.add_run()
        r.text = ln if ln else " "
        r.font.size = Pt(size)
        r.font.name = CODE_FONT
        r.font.color.rgb = CODE_HL if i in highlight else CODE_FG
    return tb


def header(slide, title, sub=None, color=NAVY):
    rect(slide, 0, 0, SW, 0.86, color)
    rect(slide, 0, 0.86, SW, 0.045, ORANGE)
    text(slide, 0.55, 0.10, 10.5, 0.66, title, size=27, color=WHITE, bold=True,
         anchor=MSO_ANCHOR.MIDDLE)
    if sub:
        text(slide, 0.58, 0.97, 12.2, 0.34, sub, size=13, color=SOFT)
    # page number
    text(slide, SW - 1.0, SH - 0.34, 0.7, 0.3, str(page_no), size=11, color=SOFT,
         align=PP_ALIGN.RIGHT)
    text(slide, 0.55, SH - 0.34, 6.0, 0.3, "vLLM main 分支 · GPU · PP 流水线并行",
         size=10, color=SOFT)


def picture_fit(slide, path, x, y, max_w, max_h):
    im = Image.open(path)
    iw, ih = im.size
    scale = min(max_w / iw, max_h / ih)
    w, h = iw * scale, ih * scale
    px = x + (max_w - w) / 2
    py = y + (max_h - h) / 2
    slide.shapes.add_picture(path, Inches(px), Inches(py), Inches(w), Inches(h))


def img_path(name):
    return os.path.join(IMG, name)


def section(part, title, sub, color):
    s = new_slide()
    rect(s, 0, 0, SW, SH, color)
    text(s, 0.9, 2.1, 3.0, 1.4, part, size=54, color=WHITE, bold=True)
    rect(s, 0.95, 3.5, 1.7, 0.06, WHITE)
    text(s, 0.95, 3.75, 11.0, 1.0, title, size=40, color=WHITE, bold=True)
    text(s, 0.95, 4.85, 11.2, 1.6, sub, size=17, color=RGBColor(0xE2, 0xE8, 0xF0),
         line_spacing=1.35)
    text(s, SW - 1.0, SH - 0.34, 0.7, 0.3, str(page_no), size=11,
         color=RGBColor(0xCB, 0xD5, 0xE0), align=PP_ALIGN.RIGHT)
    return s


# ================================================================ 1. cover
s = new_slide()
rect(s, 0, 0, SW, SH, NAVY)
rect(s, 0, 4.62, SW, 0.05, ORANGE)
text(s, 0.9, 1.45, 11.5, 0.5, "vLLM main 分支 · GPU 实现深度解析", size=18,
     color=RGBColor(0xF6, 0xAD, 0x55), bold=True)
text(s, 0.9, 2.0, 11.8, 1.6, "PP 流水线并行\n(Pipeline Parallelism)", size=48,
     color=WHITE, bold=True, line_spacing=1.15)
text(s, 0.9, 4.85, 11.6, 1.5, [
    [("从“工厂流水线”到源码:概念 → 机制 → vLLM V1 GPU 实现 → 实战选型",
      {"size": 20, "color": WHITE})],
    [("由浅入深,零基础也能看懂", {"size": 15, "color": RGBColor(0xCB, 0xD5, 0xE0)})],
], line_spacing=1.5)
text(s, 0.9, 6.55, 11.5, 0.4,
     "代码基准:vllm 仓库 main 分支(commit 70c0016)· 仅分析 NVIDIA GPU 路径",
     size=12, color=RGBColor(0xA0, 0xAE, 0xC0))

# ================================================================ 2. agenda
s = new_slide()
header(s, "目录:四个部分,由浅入深")
cards = [
    ("第一部分", "认识 PP", "为什么需要并行?\nPP 是什么?和 TP 有什么区别?\n一次前向传播是怎么“接力”的?", BLUE),
    ("第二部分", "三个关键问题", "① 模型的层怎么切?\n② 阶段之间怎么传数据?\n③ 流水线“气泡”怎么消除?", ORANGE),
    ("第三部分", "vLLM 源码实现", "进程组怎么划分?\nWorker 每一步做什么?\n核心代码逐行走读(GPU 路径)", TEAL),
    ("第四部分", "实战与选型", "命令行怎么开 PP?\n什么时候该用 PP?\n有哪些限制和坑?", PURPLE),
]
cx = 0.55
for i, (tag, t, body, color) in enumerate(cards):
    x = cx + i * 3.13
    rrect(s, x, 1.55, 2.88, 4.9, LIGHT_BG, line=color)
    rect(s, x, 1.55, 2.88, 0.14, color)
    text(s, x + 0.22, 1.88, 2.4, 0.4, tag, size=14, color=color, bold=True)
    text(s, x + 0.22, 2.28, 2.4, 0.6, t, size=21, color=DARK, bold=True)
    text(s, x + 0.22, 3.05, 2.5, 3.2, body, size=13, color=SOFT, line_spacing=1.45)

# ================================================================ 3. section 1
section("PART 1", "认识 PP 流水线并行", [
    [("先建立直觉:", {"bold": True, "size": 19})],
    [("• 为什么单张 GPU 不够用,需要“并行”?", {})],
    [("• PP 把模型按“层”切成几段,每张 GPU 只负责一段", {})],
    [("• 数据像工厂流水线一样,在各 GPU 之间接力传递", {})],
], BLUE)

# ================================================================ 4. why parallel
s = new_slide()
header(s, "为什么需要并行?单卡的两个瓶颈", "第一部分 · 认识 PP")
bullets(s, 0.55, 1.35, 6.4, 2.6, [
    (0, [("瓶颈 ①:显存放不下", {"bold": True, "color": RED})]),
    (1, "模型权重几十~几百 GB,还有 KV Cache;单卡 80GB 也可能装不下"),
    (0, [("瓶颈 ②:算得不够快", {"bold": True, "color": RED})]),
    (1, "一张卡的算力有限,吞吐和延迟都有天花板"),
    (0, [("于是有了三种并行策略,各解决不同问题", {"bold": True})]),
], size=16)
rows = [
    ("DP 数据并行", "复制整个模型,每个副本跑不同请求", "提升整体吞吐", "显存无节省(每卡一份完整模型)", GREEN),
    ("TP 张量并行", "把每一层的矩阵切到多卡,一起算", "单层/整卡放不下;降低延迟", "每层都要通信,需要 NVLink,限节点内", BLUE),
    ("PP 流水线并行", "把模型按层切段,每卡负责一段", "层数太多/跨节点放不下", "有流水线气泡,单请求延迟变高", ORANGE),
]
y = 3.5
col_x = [0.55, 3.15, 6.85, 10.05]
col_w = [2.5, 3.6, 3.1, 2.75]
heads = ["策略", "怎么切", "解决什么", "代价"]
for cx_, cw, htxt in zip(col_x, col_w, heads):
    rect(s, cx_, y, cw - 0.12, 0.46, NAVY)
    text(s, cx_ + 0.12, y + 0.05, cw - 0.3, 0.36, htxt, size=14, color=WHITE, bold=True)
y += 0.52
for name, how, good, bad, color in rows:
    hrow = 0.84
    rrect(s, col_x[0], y, col_w[0] - 0.12, hrow, LIGHT_BG, line=color)
    text(s, col_x[0] + 0.12, y + 0.1, col_w[0] - 0.3, 0.7, name, size=14, color=color, bold=True)
    for cx_, cw, txt in zip(col_x[1:], col_w[1:], [how, good, bad]):
        rrect(s, cx_, y, cw - 0.12, hrow, LIGHT_BG, line=color)
        text(s, cx_ + 0.12, y + 0.07, cw - 0.34, hrow - 0.12, txt, size=12, color=DARK,
             line_spacing=1.12)
    y += hrow + 0.08
text(s, 0.55, y + 0.04, 12.3, 0.4, [
    [("本 PPT 的主角是 ", {"size": 14, "color": SOFT}),
     ("PP", {"size": 16, "color": ORANGE, "bold": True}),
     (":大模型跨节点部署的关键,也是理解 vLLM 分布式执行的最好切入点", {"size": 14, "color": SOFT})],
])

# ================================================================ 5. concept image
s = new_slide()
header(s, "PP 是什么?按“层”切模型,每张 GPU 负责一段", "第一部分 · 认识 PP")
picture_fit(s, img_path("pp-01-concept.png"), 0.45, 1.38, 12.45, 5.72)

# ================================================================ 6. dataflow image
s = new_slide()
header(s, "一次前向传播:隐状态在各阶段之间“接力”", "第一部分 · 认识 PP")
picture_fit(s, img_path("pp-02-dataflow.png"), 0.45, 1.38, 12.45, 5.75)

# ================================================================ 7. TP vs PP
s = new_slide()
header(s, "PP 与 TP 的分工:什么时候用哪个?", "第一部分 · 认识 PP")
picture_fit(s, img_path("pp-08-compare.png"), 0.45, 1.38, 12.45, 5.75)

# ================================================================ 8. section 2
section("PART 2", "三个关键问题", [
    [("要把 PP 做出来,必须回答三个问题:", {"bold": True, "size": 19})],
    [("① 层怎么切? —— 切得均不均,直接决定有没有 GPU 在“等”", {})],
    [("② 阶段间怎么传? —— 隐状态如何在 GPU 之间高效搬运", {})],
    [("③ 气泡怎么填? —— 怎么让所有 GPU 一直有活干", {})],
], ORANGE)

# ================================================================ 9. layer split
s = new_slide()
header(s, "问题① 层怎么切?尽量均衡,余数照顾中间段", "第二部分 · 关键问题")
picture_fit(s, img_path("pp-07-split.png"), 0.45, 1.38, 12.45, 5.75)

# ================================================================ 10. communication protocol
s = new_slide()
header(s, "问题② 阶段间怎么传?元数据走 CPU,张量走 GPU", "第二部分 · 关键问题")
bullets(s, 0.55, 1.3, 6.55, 4.0, [
    (0, [("传什么:", {"bold": True})]),
    (1, [("IntermediateTensors", {"font": CODE_FONT, "size": 14, "color": PURPLE}),
         (" = {hidden_states, residual} 两个张量", {})]),
    (0, [("两步协议(parallel_state.py):", {"bold": True})]),
    (1, [("先发元数据:", {"bold": True}),
         ("send_object 走 CPU(gloo),告诉对方几个张量、什么形状/dtype", {})]),
    (1, [("再传张量本体:", {"bold": True}),
         ("isend/irecv 走 GPU(NCCL 点对点),异步不阻塞", {})]),
    (0, [("非阻塞设计:", {"bold": True})]),
    (1, "发送方:handle 存起来,下一轮开头才 wait"),
    (1, [("接收方:包成 ", {}),
         ("AsyncIntermediateTensors", {"font": CODE_FONT, "size": 13, "color": PURPLE}),
         (",访问 .tensors 时才真正 wait(懒同步)", {})]),
    (0, [("TP>1 的优化:只发本 TP rank 的切片,接收方 TP 组内 all-gather 还原", {"bold": True})]),
], size=15, gap=5)
code_block(s, 7.35, 1.35, 5.45, 4.1, [
    "# parallel_state.py · isend_tensor_dict",
    "metadata_list, tensor_list = _split(tensor_dict)",
    "self.send_object(metadata_list, dst)   # gloo/CPU",
    "for key, tensor in zip(keys, tensor_list):",
    "    if use_all_gather:  # TP>1 优化",
    "        tensor = tensor.reshape(ag, -1)[rank]",
    "    h = torch.distributed.isend(",
    "        tensor, dst=ranks[dst],       # rank+1",
    "        group=device_group)           # NCCL",
    "    handles.append(h)   # 非阻塞!",
    "return handles",
    "",
    "# 接收方 gpu_worker.py",
    "tensor_dict, handles, post = \\",
    "    get_pp_group().irecv_tensor_dict(...)",
    "it = AsyncIntermediateTensors(  # 懒同步",
    "    tensor_dict, comm_handles=handles)",
], size=12, title="阶段间通信:两级协议 + 非阻塞")
text(s, 0.55, 5.75, 12.3, 1.3, [
    [("采样结果怎么回来? ", {"size": 16, "bold": True, "color": PURPLE}),
     ("末阶段采样后,用 ", {"size": 15}),
     ("pp_broadcast", {"font": CODE_FONT, "size": 14, "color": PURPLE, "bold": True}),
     (" 把 sampled_token_ids 广播给所有阶段(vllm/v1/worker/gpu/pp_utils.py);"
      "非末阶段 pp_receive 后 postprocess 同步本地状态 —— 因为每个阶段都要管理自己那几层的 KV Cache",
      {"size": 15})],
], line_spacing=1.3)

# ================================================================ 11. bubble
s = new_slide()
header(s, "问题③ 流水线“气泡”:多批次错开执行来填满", "第二部分 · 关键问题")
picture_fit(s, img_path("pp-03-bubble.png"), 0.45, 1.38, 12.45, 5.3)
text(s, 0.55, 6.85, 12.3, 0.4, [
    [("vLLM main 的做法:EngineCore 里维护一个 ", {"size": 13, "color": SOFT}),
     ("batch queue(深度 = pp_size)", {"size": 13, "color": ORANGE, "bold": True}),
     (",队列满了才回头收最老批次的结果", {"size": 13, "color": SOFT})],
])

# ================================================================ 12. section 3
section("PART 3", "vLLM main 源码实现(GPU)", [
    [("带着三个问题看代码:", {"bold": True, "size": 19})],
    [("• 进程组:parallel_state.initialize_model_parallel()", {})],
    [("• 执行流:gpu_worker.execute_model() 的收发与计算", {})],
    [("• 填充:engine/core.py 的 batch queue", {})],
    [("• 模型侧:SupportsPP 接口 + Llama 的真实实现", {})],
], TEAL)

# ================================================================ 13. arch
s = new_slide()
header(s, "总览:PP 在 vLLM V1 引擎中的位置", "第三部分 · 源码实现")
picture_fit(s, img_path("pp-05-arch.png"), 0.45, 1.38, 12.45, 5.72)

# ================================================================ 14. ranks
s = new_slide()
header(s, "进程组怎么划分?rank 布局 [DP × PP × PCP × TP]", "第三部分 · 源码实现")
picture_fit(s, img_path("pp-04-ranks.png"), 0.45, 1.38, 12.45, 5.75)

# ================================================================ 15. worker steps
s = new_slide()
header(s, "每个 Worker 每步做什么?核心四步", "第三部分 · 源码实现")
picture_fit(s, img_path("pp-06-exec.png"), 0.45, 1.38, 12.45, 5.1)
text(s, 0.55, 6.6, 12.3, 0.55, [
    [("关键:收发都是", {"size": 14, "color": SOFT}),
     ("非阻塞", {"size": 14, "color": TEAL, "bold": True}),
     ("的 —— 上一阶段还在算,下一阶段已经在收隐状态;本阶段算完立刻发,不等对方确认", {"size": 14, "color": SOFT})],
])

# ================================================================ 16. code: groups
s = new_slide()
header(s, "代码走读①:进程组初始化(parallel_state.py)", "第三部分 · 源码实现")
code_block(s, 0.55, 1.3, 7.1, 5.0, [
    "# vllm/distributed/parallel_state.py",
    "def initialize_model_parallel(",
    "        tensor_model_parallel_size=1,",
    "        pipeline_model_parallel_size=1, ...):",
    "    # rank 布局:ExternalDP × DP × PP × PCP × TP",
    "    all_ranks = torch.arange(world_size).reshape(",
    "        -1, data_parallel_size,",
    "        pipeline_model_parallel_size,",
    "        prefill_context_model_parallel_size,",
    "        tensor_model_parallel_size)",
    "",
    "    # 建 TP 组:最后一维直接切开",
    "    _TP = init_model_parallel_group(",
    "        all_ranks.view(-1, tp).unbind(0), ...)",
    "",
    "    # 建 PP 组:把 PP 维换到最后再切",
    "    group_ranks = all_ranks.transpose(2, 4)\\",
    "        .reshape(-1, pp_size).unbind(0)",
    "    _PP = init_model_parallel_group(group_ranks,",
    "        get_world_group().local_rank, backend,",
    "        group_name=\"pp\")   # ← get_pp_group()",
], size=12.5, title="vllm/distributed/parallel_state.py · initialize_model_parallel",
   highlight={20})
bullets(s, 7.95, 1.45, 5.0, 4.9, [
    (0, [("要点 1:布局顺序 [DP × PP × PCP × TP]", {"bold": True})]),
    (1, "同一张卡同时属于一个 TP 组和一个 PP 组,两个维度正交"),
    (0, [("要点 2:PP 组内 rank 顺序 = 阶段顺序", {"bold": True})]),
    (1, [("rank+1 是下一阶段(isend 目标),rank-1 是上一阶段(irecv 来源)", {})]),
    (1, [("is_first_rank / is_last_rank", {"font": CODE_FONT, "size": 13, "color": TEAL}),
         (" 标记首末阶段", {})]),
    (0, [("要点 3:相邻 rank 尽量同机", {"bold": True})]),
    (1, "docstring 建议:TP 组放同一台机器(NVLink),PP 跨机"),
    (0, [("world_size = PP × TP × PCP(每个 DP 副本内)", {"bold": True})]),
], size=14, gap=5)

# ================================================================ 17. code: model side
s = new_slide()
header(s, "代码走读②:模型怎么配合?SupportsPP + Llama 实现", "第三部分 · 源码实现")
code_block(s, 0.55, 1.3, 6.6, 5.35, [
    "# model_executor/models/llama.py",
    "class LlamaForCausalLM(nn.Module, SupportsPP):",
    "",
    "    # ① 首阶段才有 Embedding",
    "    if get_pp_group().is_first_rank:",
    "        self.embed_tokens = VocabParallelEmbedding(...)",
    "    else:",
    "        self.embed_tokens = PPMissingLayer()  # 占位",
    "",
    "    # ② 只建自己那几层,其余用占位符",
    "    self.start_layer, self.end_layer, self.layers = \\",
    "        make_layers(config.num_hidden_layers, ...)",
    "",
    "    # ③ 末阶段才有 Norm 和 LM Head",
    "    if get_pp_group().is_last_rank:",
    "        self.norm = RMSNorm(...)",
    "        self.lm_head = ParallelLMHead(...)",
    "",
    "    # ④ forward:非首阶段吃 intermediate_tensors,",
    "    #    非末阶段吐 intermediate_tensors",
    "    def forward(self, input_ids, positions,",
    "            intermediate_tensors=None, ...):",
    "        if not get_pp_group().is_first_rank:",
    "            hidden = intermediate_tensors[\"hidden_states\"]",
    "            residual = intermediate_tensors[\"residual\"]",
], size=11.5, title="Llama 的 PP 改造:四处关键改动")
bullets(s, 7.4, 1.45, 5.5, 5.1, [
    (0, [("SupportsPP 接口(models/interfaces.py):", {"bold": True})]),
    (1, [("make_empty_intermediate_tensors()", {"font": CODE_FONT, "size": 12, "color": TEAL}),
         (" —— 供显存预估(profiling)用", {})]),
    (1, [("forward(..., intermediate_tensors)", {"font": CODE_FONT, "size": 12, "color": TEAL}),
         (" —— 吃上一段、吐给下一段", {})]),
    (0, [("make_layers(models/utils.py):", {"bold": True})]),
    (1, [("按 get_pp_indices 算出 [start, end),只真正构建这段层;"
          "范围外的层用 ", {}),
         ("PPMissingLayer", {"font": CODE_FONT, "size": 12, "color": TEAL}),
         (" 占位(不分配权重)", {})]),
    (0, [("权重加载自动跳过别的阶段:", {"bold": True})]),
    (1, [("is_pp_missing_parameter()", {"font": CODE_FONT, "size": 12, "color": TEAL}),
         (" —— 每卡只从磁盘读自己那几层的权重,省显存也省加载时间", {})]),
    (0, [("tie_word_embeddings 的模型:末阶段也保留 embed_tokens", {"bold": True})]),
], size=13.5, gap=5)

# ================================================================ 18. code: worker
s = new_slide()
header(s, "代码走读③:Worker 的收发主循环(gpu_worker.py)", "第三部分 · 源码实现")
code_block(s, 0.55, 1.3, 7.35, 5.35, [
    "# vllm/v1/worker/gpu_worker.py · execute_model",
    "def execute_model(self, scheduler_output):",
    "    # ① 等上一轮的异步发送彻底落账",
    "    for handle in self._pp_send_work:",
    "        handle.wait()",
    "",
    "    # ② 非首阶段:异步接收上一阶段发来的隐状态",
    "    if forward_pass and not get_pp_group().is_first_rank:",
    "        tensor_dict, handles, post = \\",
    "            get_pp_group().irecv_tensor_dict(",
    "                all_gather_group=get_tp_group(), ...)",
    "        intermediate_tensors = AsyncIntermediateTensors(",
    "            tensor_dict, comm_handles=handles, ...)",
    "",
    "    # ③ 跑本阶段的层(可与通信重叠)",
    "    output = self.model_runner.execute_model(",
    "        scheduler_output, intermediate_tensors)",
    "    if isinstance(output, ModelRunnerOutput | NoneType):",
    "        return output          # 首/末阶段或空批次",
    "",
    "    # ④ 非末阶段:把隐状态异步发给下一阶段",
    "    self._pp_send_work = get_pp_group().isend_tensor_dict(",
    "        output.tensors, all_gather_group=get_tp_group())",
    "    return None",
], size=12, title="每个 Worker 进程每步执行(所有阶段跑同一份代码)",
   highlight={3, 8, 22})
bullets(s, 8.2, 1.45, 4.65, 5.1, [
    (0, [("同一份代码,四种行为:", {"bold": True})]),
    (1, "首阶段:只 ③④(不收)"),
    (1, "中间阶段:①②③④ 全做"),
    (1, "末阶段:只 ①②③,产出 hidden_states 去采样"),
    (0, [("① 为什么先等发送?", {"bold": True})]),
    (1, "上一轮 isend 用的是本 rank 的发送缓冲区,必须确认对方收完才能覆写"),
    (0, [("③ 为什么能与通信重叠?", {"bold": True})]),
    (1, "irecv 已把张量挂到 NCCL 流上;模型第一层用到隐状态时才 wait(懒同步)"),
    (0, [("采样后的回传见 pp_utils.py:末阶段 broadcast,其他阶段 receive + postprocess", {"bold": True})]),
], size=13.5, gap=5)

# ================================================================ 19. code: batch queue
s = new_slide()
header(s, "代码走读④:batch queue —— 消除气泡的调度循环", "第三部分 · 源码实现")
code_block(s, 0.55, 1.3, 7.35, 5.35, [
    "# vllm/v1/executor/multiproc_executor.py",
    "@cached_property",
    "def max_concurrent_batches(self):",
    "    # PP 需要 pp_size 个并发批次填满流水线",
    "    return pp_size if pp_size > 1 else \\",
    "        (2 if async_scheduling else 1)",
    "",
    "# vllm/v1/engine/core.py · step_with_batch_queue",
    "def step_with_batch_queue(self):",
    "    # ① 队列没满就持续调度+提交(非阻塞 future)",
    "    if self.scheduler.has_requests():",
    "        scheduler_output = self.scheduler.schedule()",
    "        exec_future = self.model_executor.execute_model(",
    "            scheduler_output, non_block=True)",
    "        ...",
    "        batch_queue.appendleft((future, sched_out, ...))",
    "        if len(batch_queue) < self.batch_queue_size:",
    "            return None, True   # 继续填,不阻塞!",
    "    # ② 队列满了,才取最老批次的结果更新调度器",
    "    future, scheduler_output, _ = batch_queue.pop()",
    "    model_output = future.result()",
    "    self.scheduler.update_from_output(...)",
], size=12, title="填队列优先,收结果靠后 —— 流水线因此不空转",
   highlight={5, 11, 19})
bullets(s, 8.2, 1.45, 4.65, 5.1, [
    (0, [("为什么深度恰好是 pp_size?", {"bold": True})]),
    (1, "第 k 个批次在阶段 0 开算时,前面 k-1 个批次正排在后续阶段上;深度=阶段数 ⇒ 稳态时每个阶段各算一个批次"),
    (0, [("每个阶段拿到同一份 SchedulerOutput", {"bold": True})]),
    (1, "Executor 把它广播给所有 Worker;各阶段据此管理自己的 KV Cache"),
    (0, [("结果只从 output_rank 回收", {"bold": True})]),
    (1, [("= 末阶段的 TP rank0(world_size − tp_size)", {})]),
    (0, [("async_scheduling 时 pp=1 也用深度 2 的队列", {"bold": True})]),
], size=13.5, gap=6)

# ================================================================ 20. cuda graph + SP
s = new_slide()
header(s, "进阶:PP 与 CUDA Graph、序列并行(SP)的配合", "第三部分 · 源码实现")
bullets(s, 0.55, 1.35, 6.4, 4.6, [
    (0, [("PP × CUDA Graph(cudagraph_utils.py)", {"bold": True, "color": TEAL})]),
    (1, "FULL 模式 capture 时,非首阶段把 intermediate_tensors 当作图的固定输入 buffer"),
    (1, "非末阶段把图输出拷进 persistent buffer,重放后返回给 worker 去 isend"),
    (1, "⇒ 每个阶段的图只含自己那几层,PP 不破坏 CUDA Graph 加速"),
    (0, [("PP × SP × TP(gpu_worker.py)", {"bold": True, "color": TEAL})]),
    (1, "开启 SP 时 residual 在 TP 组内是切开的,不能整块 all-gather"),
    (1, [("worker 算出 ", {}),
         ("all_gather_tensors={\"residual\": False}", {"font": CODE_FONT, "size": 12, "color": TEAL}),
         (" 逐张量控制收发方式", {})]),
    (0, [("默认优化(TP>1):", {"bold": True, "color": TEAL})]),
    (1, "隐状态在 TP 组内是复制的 → 每个 TP rank 只发 1/TP 切片,接收方 all-gather 还原,点对点通信量降为 1/TP"),
], size=14, gap=6)
code_block(s, 7.25, 1.35, 5.55, 4.2, [
    "# cudagraph_utils.py · capture",
    "if not self.is_first_pp_rank:",
    "    model_inputs[\"input_ids\"] = None",
    "    model_inputs[\"intermediate_tensors\"] = \\",
    "        intermediate_tensors[:num_tokens]",
    "...",
    "if not self.is_last_pp_rank:",
    "    # 非末阶段:图输出写回 persistent buffer",
    "    for k, v in output.tensors.items():",
    "        self.intermediate_tensors[k][:n] = v",
    "",
    "# gpu_worker.py · SP 感知",
    "if pp > 1 and pass_config.enable_sp:",
    "    all_gather_tensors = {\"residual\":",
    "        not is_residual_scattered_for_sp(",
    "            vllm_config, num_tokens)}",
    "# SP 要求 full-graph 编译;残差被 TP 切开",
    "# 时不允许 all-gather 优化",
], size=11.5, title="两个容易忽略的配合点")
text(s, 0.55, 6.15, 12.3, 0.9, [
    [("小结:", {"size": 15, "bold": True}),
     ("PP 是“正交”的并行维度 —— 它与 CUDA Graph、SP/TP、DP 都能组合;"
      "vLLM 在每个组合点上都有专门的处理代码", {"size": 14, "color": SOFT})],
])

# ================================================================ 21. KV cache & weights
s = new_slide()
header(s, "隐形的收益:KV Cache 与权重也按阶段分摊", "第三部分 · 源码实现")
bullets(s, 0.55, 1.35, 6.3, 4.9, [
    (0, [("KV Cache 按层分摊", {"bold": True, "color": PURPLE})]),
    (1, "每个阶段只为自己持有的注意力层分配 KV block(未持有的层是 PPMissingLayer,不进 kv_cache_spec)"),
    (1, [("⇒ PP=4 时,每卡 KV Cache 显存约为 1/4 —— ", {}),
         ("长上下文场景的关键收益", {"bold": True, "color": PURPLE})]),
    (0, [("权重按层分摊", {"bold": True, "color": PURPLE})]),
    (1, "加载时 is_pp_missing_parameter 跳过其他阶段的参数,磁盘 IO 与显存都省"),
    (0, [("显存预估(profiling)也是分阶段的", {"bold": True, "color": PURPLE})]),
    (1, [("非首阶段用 ", {}),
         ("make_empty_intermediate_tensors()", {"font": CODE_FONT, "size": 12, "color": TEAL}),
         (" 造全零隐状态跑 dummy run,量出本阶段峰值显存", {})]),
    (0, [("多模态模型的 encoder 只在首阶段", {"bold": True, "color": PURPLE})]),
    (1, "encoder_cache 仅 is_first_pp_rank 创建;图片/音频只在流水线入口编码一次"),
], size=14, gap=6)
rrect(s, 7.15, 1.5, 5.65, 4.7, LIGHT_BG, line=PURPLE)
text(s, 7.45, 1.75, 5.1, 0.5, "一个 PP=4 的显存账(每卡)", size=17, color=PURPLE, bold=True)
rows = [
    ("模型权重", "全部", "约 1/4(本阶段的层)"),
    ("KV Cache", "全部层", "约 1/4(本阶段的层)"),
    ("激活/隐状态", "整模型", "本阶段 + 通信 buffer"),
    ("Embedding/ LMHead", "1 份", "只在首/末阶段"),
]
y = 2.35
heads = ["显存项", "单卡部署", "PP=4 每卡"]
cw = [1.7, 1.55, 2.15]
x = 7.45
for cx_, cw_, htxt in zip([x, x + 1.8, x + 3.45], cw, heads):
    rect(s, cx_, y, cw_ - 0.08, 0.44, PURPLE)
    text(s, cx_ + 0.1, y + 0.05, cw_ - 0.24, 0.34, htxt, size=13, color=WHITE, bold=True)
y += 0.5
for r in rows:
    for cx_, cw_, txt in zip([x, x + 1.8, x + 3.45], cw, r):
        rrect(s, cx_, y, cw_ - 0.08, 0.5, WHITE, line=PURPLE)
        text(s, cx_ + 0.1, y + 0.07, cw_ - 0.24, 0.38, txt, size=12, color=DARK)
    y += 0.58
text(s, 7.45, y + 0.1, 5.1, 1.0,
     "注意:PP 不减少总计算量,也不减少单请求延迟 —— 它省的是“每卡显存”,换来跨机扩展能力",
     size=13, color=SOFT, line_spacing=1.3)

# ================================================================ 22. section 4
section("PART 4", "实战与选型", [
    [("看完原理,动手之前最后一站:", {"bold": True, "size": 19})],
    [("• 一行命令怎么开 PP?多机怎么配?", {})],
    [("• 官方选型建议:什么时候 PP 优于 TP?", {})],
    [("• 限制清单:哪些特性与 PP 不兼容?", {})],
], PURPLE)

# ================================================================ 23. usage
s = new_slide()
header(s, "怎么用?一行命令 + 一个环境变量", "第四部分 · 实战与选型")
code_block(s, 0.55, 1.35, 6.6, 3.0, [
    "# 单机 8 卡:TP=4 × PP=2",
    "vllm serve <model> \\",
    "    --tensor-parallel-size 4 \\",
    "    --pipeline-parallel-size 2      # 或 -pp 2",
    "",
    "# 2 机 × 8 卡(Ray 集群):节点内 TP,节点间 PP",
    "vllm serve <model> \\",
    "    --tensor-parallel-size 8 \\",
    "    --pipeline-parallel-size 2",
    "",
    "# 手动指定每段层数(总和须等于总层数)",
    "export VLLM_PP_LAYER_PARTITION=\"8,8,9,8\"",
], size=13, title="启动命令(vllm serve / LLM 类同样适用)")
bullets(s, 0.55, 4.65, 6.6, 2.2, [
    (0, [("执行后端自动选择:", {"bold": True})]),
    (1, "PP×TP ≤ 本机 GPU 数 → 自动用 mp(多进程);跨机 → 用 Ray 集群"),
    (0, [("日志自检:", {"bold": True})]),
    (1, "切层不均时会打印 Hidden layers were unevenly partitioned: [...]"),
], size=14, gap=6)
rrect(s, 7.5, 1.35, 5.3, 5.5, LIGHT_BG, line=GREEN)
text(s, 7.8, 1.6, 4.7, 0.5, "官方选型建议(文档原文)", size=17, color=GREEN, bold=True)
bullets(s, 7.8, 2.2, 4.75, 4.4, [
    (0, "单卡放得下 → 不要并行"),
    (0, "单卡放不下、单机放得下 → 节点内 TP"),
    (0, [("单机放不下 → 节点内 TP × 节点间 PP", {"bold": True})]),
    (1, "tp = 每机 GPU 数,pp = 机器数"),
    (0, [("GPU 无 NVLink(如 L40S)→ 优先 PP 而非 TP", {"bold": True, "color": ORANGE})]),
    (1, "PP 通信量小,普通 PCIe/网络可承受"),
    (0, "卡数与层数不整除 → PP 支持不均匀切分"),
], size=14, gap=8)
text(s, 7.8, 6.35, 4.75, 0.5, "出处:docs/serving/parallelism_scaling.md",
     size=11, color=SOFT)

# ================================================================ 24. limitations
s = new_slide()
header(s, "限制与注意事项(踩坑前先看)", "第四部分 · 实战与选型")
bullets(s, 0.55, 1.35, 12.3, 5.3, [
    (0, [("① 模型必须实现 ", {"bold": True}),
         ("SupportsPP", {"font": CODE_FONT, "size": 14, "color": RED, "bold": True}),
         (" 接口", {"bold": True})]),
    (1, "启动时校验:registry.is_pp_supported_model();主流模型(Llama/Qwen/DeepSeek/Mixtral 等)均已支持,冷门模型会直接报 NotImplementedError"),
    (0, [("② EAGLE3 投机采样 × PP 不兼容", {"bold": True})]),
    (1, "EAGLE3 需要目标模型所有层的辅助隐状态,而 PP 把层切开了 → 直接 raise(model_runner.py);普通 EAGLE/MTP 可用,且只在末阶段运行 speculator"),
    (0, [("③ Elastic EP(弹性专家并行)× PP 不兼容", {"bold": True})]),
    (1, "ParallelConfig.__post_init__ 直接报错;MoE 模型用普通 EP + PP 组合不受影响"),
    (0, [("④ 单请求延迟升高", {"bold": True})]),
    (1, "一个 token 要顺序流过 pp_size 个阶段;延迟敏感、并发低的场景,PP 不是好选择(优先 TP/单卡)"),
    (0, [("⑤ 同一 DP 组内的 generate 必须一起调用", {"bold": True})]),
    (1, "否则 PP 组内点对点通信会死锁(initialize_model_parallel 的 docstring 明确警告)"),
    (0, [("⑥ PP 不省通信总量以外的任何东西:不减少计算量,批太小填不满流水线时收益有限", {"bold": True})]),
], size=14.5, gap=7)

# ================================================================ 25. summary
s = new_slide()
header(s, "总结:一张源码地图带走 PP", "第四部分 · 总结")
bullets(s, 0.55, 1.3, 5.9, 2.9, [
    (0, [("一句话:", {"bold": True, "color": NAVY}),
         ("PP = 按层切段 + 隐状态接力 + 多批次填充", {"bold": True, "color": ORANGE, "size": 17})]),
    (0, [("三个关键机制 ↔ 三处代码:", {"bold": True})]),
    (1, "切层 → get_pp_indices(均衡 + 余数给中间段)"),
    (1, "通信 → isend/irecv 两级协议(元数据 gloo + 张量 NCCL)"),
    (1, "填充 → batch queue 深度 = pp_size"),
    (0, [("选型口诀:", {"bold": True})]),
    (1, "节点内用 TP,跨节点 TP×PP;无 NVLink 优先 PP"),
], size=14.5, gap=6)
rows = [
    ("vllm/config/parallel.py", "pipeline_parallel_size 参数与校验"),
    ("vllm/distributed/parallel_state.py", "PP 进程组 · send/recv_tensor_dict"),
    ("vllm/distributed/utils.py", "get_pp_indices 层切分算法"),
    ("vllm/model_executor/models/interfaces.py", "SupportsPP 协议"),
    ("vllm/model_executor/models/utils.py", "make_layers · PPMissingLayer"),
    ("vllm/v1/worker/gpu_worker.py", "execute_model 收发主循环"),
    ("vllm/v1/worker/gpu/pp_utils.py", "pp_broadcast / pp_receive"),
    ("vllm/v1/engine/core.py", "batch queue 填充流水线"),
    ("vllm/v1/executor/multiproc_executor.py", "max_concurrent_batches · 广播"),
]
y = 1.35
x = 6.75
rect(s, x, y, 6.05, 0.42, NAVY)
text(s, x + 0.14, y + 0.05, 3.6, 0.32, "源码文件(main 分支)", size=13, color=WHITE, bold=True)
text(s, x + 3.85, y + 0.05, 2.1, 0.32, "管什么", size=13, color=WHITE, bold=True)
y += 0.48
for f, desc in rows:
    rrect(s, x, y, 6.05, 0.47, LIGHT_BG, line=RGBColor(0xCB, 0xD5, 0xE0))
    text(s, x + 0.14, y + 0.06, 3.75, 0.36, f, size=11.5, color=DARK, font_name=CODE_FONT)
    text(s, x + 3.85, y + 0.06, 2.15, 0.36, desc, size=11, color=SOFT)
    y += 0.53

# ================================================================ 26. closing
s = new_slide()
rect(s, 0, 0, SW, SH, NAVY)
text(s, 0.9, 2.7, 11.5, 1.2, "谢谢观看", size=48, color=WHITE, bold=True)
text(s, 0.9, 4.0, 11.5, 1.2, [
    [("下一步建议:对照本 PPT 的源码地图,在 vllm 仓里跟一遍 gpu_worker.execute_model", {"size": 17})],
    [("配图与脚本:assets/vllm-novidia/(make-pp-ppt-diagrams.py · build-pp-ppt.py)", {"size": 13, "color": RGBColor(0xA0, 0xAE, 0xC0)})],
], line_spacing=1.6)

prs.save(OUT)
print("saved:", OUT, "| slides:", len(prs.slides.__iter__.__self__._sldIdLst))
