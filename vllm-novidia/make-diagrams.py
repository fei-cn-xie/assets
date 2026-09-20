#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate standalone SVG architecture diagrams for the vLLM main-branch report.

Run:  python make-diagrams.py
Output: img/*.svg  (subdirectory next to this script)

纯标准库实现，不依赖 graphviz / mermaid / 浏览器，输出可直接用浏览器或
VS Code / Markdown 预览打开。
"""

from __future__ import annotations

import os
import sys
import html

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IMG_DIRNAME = "img"

FONT = "Microsoft YaHei, PingFang SC, Segoe UI, sans-serif"

PALETTE = {
    "client": ("#E8F1FB", "#2B6CB0"),
    "entry": ("#EAF7EE", "#2F855A"),
    "engine": ("#FFF4E6", "#C05621"),
    "core": ("#FDECEC", "#C53030"),
    "exec": ("#F0EBFF", "#553C9A"),
    "model": ("#E6FFFA", "#2C7A7B"),
    "kernel": ("#FFF0F6", "#B83280"),
    "support": ("#F2F4F7", "#4A5568"),
    "store": ("#FFFBEA", "#B7791F"),
}


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def est_width(text: str, fs: float) -> float:
    """Rough advance width: CJK ~= 1.0em, latin ~= 0.55em."""
    total = 0.0
    for ch in text:
        total += fs if ord(ch) > 0x2E80 else fs * 0.55
    return total


WARNINGS: list[str] = []
RECTS: list[tuple[float, float, float, float, str]] = []


def check_overlaps() -> None:
    """Report element-overlapping boxes (bands and arrows are not tracked)."""
    for i in range(len(RECTS)):
        x1, y1, w1, h1, t1 = RECTS[i]
        for j in range(i + 1, len(RECTS)):
            x2, y2, w2, h2, t2 = RECTS[j]
            if x1 < x2 + w2 and x2 < x1 + w1 and y1 < y2 + h2 and y2 < y1 + h1:
                WARNINGS.append(f"box overlap: '{t1}' {[x1, y1, w1, h1]} vs "
                                f"'{t2}' {[x2, y2, w2, h2]}")


def box(x, y, w, h, lines, kind="engine", fs=13, radius=8, lh=17):
    fill, stroke = PALETTE[kind]
    if isinstance(lines, str):
        lines = [lines]
    RECTS.append((x, y, w, h, lines[0]))
    out = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>'
    ]
    total = len(lines) * lh
    start = y + (h - total) / 2 + lh - 4
    for i, line in enumerate(lines):
        weight = "600" if i == 0 else "400"
        size = fs if i == 0 else fs - 1
        color = stroke if i == 0 else "#3D4852"
        if est_width(line, size) > w - 14:
            WARNINGS.append(
                f"overflow: box({x},{y},{w}x{h}) line='{line}' "
                f"needs~{est_width(line, size):.0f}px"
            )
        if start + (len(lines) - 1) * lh > y + h - 6:
            WARNINGS.append(f"vertical overflow: box({x},{y},{w}x{h})")
        out.append(
            f'<text x="{x + w / 2:.0f}" y="{start + i * lh:.0f}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
            f'fill="{color}">{esc(line)}</text>'
        )
    return "\n".join(out)


def band(x, y, w, h, title, color="#4A5568"):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="none" '
        f'stroke="{color}" stroke-width="1.4" stroke-dasharray="6 4"/>'
        f'<text x="{x + 14}" y="{y + 21}" font-family="{FONT}" font-size="15" '
        f'font-weight="700" fill="{color}">{esc(title)}</text>'
    )


def arrow(x1, y1, x2, y2, label=None, dashed=False, color="#4A5568", lx=None, ly=None):
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    out = [
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="1.8"{dash} marker-end="url(#arrow)"/>'
    ]
    if label:
        mx = lx if lx is not None else (x1 + x2) / 2
        my = ly if ly is not None else (y1 + y2) / 2 - 6
        out.append(
            f'<text x="{mx:.0f}" y="{my:.0f}" text-anchor="middle" font-family="{FONT}" '
            f'font-size="11.5" fill="{color}">{esc(label)}</text>'
        )
    return "\n".join(out)


def svg_open(w, h, title):
    RECTS.clear()  # per-diagram coordinate space
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"
     viewBox="0 0 {w} {h}" font-family="{FONT}">
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"
          markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#4A5568"/>
  </marker>
</defs>
<rect width="{w}" height="{h}" fill="#FFFFFF"/>
<text x="{w / 2:.0f}" y="36" text-anchor="middle" font-size="23" font-weight="700"
      fill="#1A202C">{esc(title)}</text>
'''


def svg_close():
    check_overlaps()
    return "</svg>\n"


def write(name, content):
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), IMG_DIRNAME)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(content)
    print(f"wrote {IMG_DIRNAME}/{name} ({len(content)} bytes)")


# --------------------------------------------------------------------------
# 图 1：全局分层架构
# --------------------------------------------------------------------------
def diagram_overview():
    W, H = 1320, 1000
    s = [svg_open(W, H, "vLLM main 分支：全局分层架构")]

    # 客户端
    s.append(band(30, 60, 1000, 86, "客户端"))
    clients = ["OpenAI 兼容 SDK", "Anthropic SDK", "gRPC 客户端",
               "实时音频 WebSocket", "Python 离线 API / CLI"]
    for i, c in enumerate(clients):
        s.append(box(48 + i * 196, 88, 180, 44, c, "client"))

    # 接入层
    s.append(band(30, 168, 1000, 120, "接入层  vllm/entrypoints/"))
    entries = [
        ("api_router.py", "协议解析 / 校验 / SSE 组包"),
        ("serving.py", "chat 模板 / 分词 / 采样参数"),
        ("renderers/ + tokenizers/", "渲染与分词"),
        ("offline LLM 类 + CLI", "离线与命令行入口"),
    ]
    for i, (t, sub) in enumerate(entries):
        s.append(box(48 + i * 246, 198, 226, 74, [t, sub], "entry"))

    # 引擎层
    s.append(band(30, 310, 1000, 132, "引擎层  vllm/v1/engine/"))
    eng = [
        ["EngineClient", "统一抽象 protocol.py"],
        ["AsyncLLM / LLMEngine", "在线 / 离线门面"],
        ["InputProcessor", "生成 EngineCoreRequest"],
        ["OutputProcessor", "Detokenizer 反分词"],
        ["EngineCoreClient", "Inproc / AsyncMP / DP"],
    ]
    for i, lines in enumerate(eng):
        s.append(box(48 + i * 196, 342, 180, 82, lines, "engine"))

    # 引擎核心
    s.append(band(30, 464, 1000, 132, "引擎核心  EngineCore（独立进程）"))
    core = [
        ["EngineCoreProc", "busy loop + 2 个 IO 线程"],
        ["Scheduler", "排队 / 批调度 / 结束判定"],
        ["KVCacheManager", "块分配与前缀缓存"],
        ["EncoderCacheManager", "多模态 encoder 缓存"],
        ["StructuredOutputManager", "grammar 与 bitmask"],
    ]
    for i, lines in enumerate(core):
        s.append(box(48 + i * 196, 496, 180, 82, lines, "core"))

    # 执行层 + Worker
    s.append(band(30, 618, 1000, 132, "执行层与 Worker  vllm/v1/executor/ + vllm/v1/worker/"))
    ex = [
        ["UniProcExecutor", "单卡同进程"],
        ["MultiprocExecutor", "多卡多进程 MessageQueue"],
        ["RayExecutor / V2", "跨节点"],
        ["Worker", "设备初始化 · 显存 · KV"],
        ["GPUModelRunner V1 / V2", "默认 V1，V2 需显式开启"],
    ]
    for i, lines in enumerate(ex):
        s.append(box(48 + i * 196, 650, 180, 82, lines, "exec"))

    # 模型与内核
    s.append(band(30, 772, 1000, 190, "模型、算子与内核  vllm/model_executor/ + vllm/v1/ + vllm/compilation/"))
    mk = [
        ["models/", "模型定义与注册表"],
        ["layers/", "Linear / MoE / RoPE / Pooler"],
        ["model_loader/", "权重加载"],
        ["quantization/", "量化方法"],
        ["v1/attention/", "后端选择与 paged attention"],
        ["v1/sample/ + spec_decode", "logits 处理 / 拒绝采样"],
        ["kernels/ + _custom_ops", "triton 与 C++ 算子"],
        ["compilation/", "torch.compile 与 CUDA Graph"],
    ]
    for i, lines in enumerate(mk):
        s.append(box(48 + (i % 4) * 246, 806 + (i // 4) * 78, 226, 66, lines, "model"))

    # 支撑子系统（右侧竖栏）
    s.append(band(1050, 60, 240, 902, "支撑子系统"))
    sup = [
        ["config/ + envs.py", "VllmConfig 配置树"],
        ["platforms/", "设备抽象与选择"],
        ["distributed/", "TP / PP / DP / EP + NCCL"],
        ["distributed/kv_transfer/", "P/D 分离连接器"],
        ["v1/kv_offload/", "KV 卸载与分层缓存"],
        ["v1/metrics/ + profiler/", "观测与性能模型"],
        ["lora/ + plugins/", "适配器与扩展点"],
        ["utils/ + transformers_utils/", "横切工具与 HF 适配"],
    ]
    for i, lines in enumerate(sup):
        s.append(box(1066, 88 + i * 106, 208, 88, lines, "support", fs=12))

    # 纵向主链路箭头
    for y1, y2 in ((146, 198), (288, 342), (442, 496), (596, 650), (732, 806)):
        s.append(f'<line x1="530" y1="{y1}" x2="530" y2="{y2}" stroke="#A0AEC0" '
                 f'stroke-width="2.5" marker-end="url(#arrow)"/>')
    s.append(f'<text x="546" y="176" font-size="11.5" fill="#718096">HTTP / gRPC / WS</text>')
    s.append(f'<text x="546" y="318" font-size="11.5" fill="#718096">EngineClient</text>')
    s.append(f'<text x="546" y="472" font-size="11.5" fill="#718096">ZMQ msgpack</text>')
    s.append(f'<text x="546" y="626" font-size="11.5" fill="#718096">execute_model</text>')
    s.append(f'<text x="546" y="782" font-size="11.5" fill="#718096">model forward</text>')
    s.append(f'<line x1="1032" y1="530" x2="1064" y2="530" stroke="#A0AEC0" '
             f'stroke-width="1.6" stroke-dasharray="5 4"/>')

    s.append(svg_close())
    write("architecture-overview.svg", "\n".join(s))


# --------------------------------------------------------------------------
# 图 2：进程 / 线程 / 通信拓扑
# --------------------------------------------------------------------------
def diagram_process():
    W, H = 1320, 760
    s = [svg_open(W, H, "vLLM 进程、线程与通信拓扑")]

    # 进程 1
    s.append(band(30, 60, 380, 620, "进程 1：API Server（asyncio + uvicorn）", "#2B6CB0"))
    p1 = [
        ["uvicorn 事件循环", "FastAPI 路由 + 中间件"],
        ["AsyncLLM", "引擎门面"],
        ["OutputProcessor / Detokenizer", "组包 RequestOutput"],
        ["output_handler 后台 asyncio 任务", "消费 EngineCoreOutputs"],
        ["EngineCoreClient", "ROUTER + PULL 套接字"],
    ]
    for i, lines in enumerate(p1):
        s.append(box(48, 96 + i * 112, 344, 88, lines, "client"))

    # 进程 2
    s.append(band(470, 60, 380, 620, "进程 2：EngineCore（DP 时每 rank 一个）", "#C53030"))
    p2 = [
        ["IO 线程 process_input_sockets", "解帧 / ADD 预处理"],
        ["IO 线程 process_output_sockets", "EngineCoreOutputs 回传"],
        ["主线程 run_busy_loop", "step() 循环"],
        ["Scheduler + KVCacheManager", "排队与 KV 分配"],
        ["input_queue / aborts_queue / output_queue", "线程间解耦"],
    ]
    for i, lines in enumerate(p2):
        s.append(box(488, 96 + i * 112, 344, 88, lines, "core"))

    # 进程 3
    s.append(band(910, 60, 380, 620, "进程 3..N：Worker（每 rank 一个）", "#553C9A"))
    p3 = [
        ["WorkerProc.worker_main", "子进程入口"],
        ["GPUModelRunner V1 / V2", "前向与采样"],
        ["CUDA context + 权重分片", "设备资源"],
        ["KV cache 张量", "按 KVCacheConfig 分配"],
        ["NCCL 通信组", "TP / PP / DP / EP"],
    ]
    for i, lines in enumerate(p3):
        s.append(box(928, 96 + i * 112, 344, 88, lines, "exec"))

    # 通道
    s.append(arrow(410, 150, 470, 150, "ADD / ABORT / UTILITY", color="#2B6CB0",
                   lx=440, ly=140))
    s.append(arrow(470, 262, 410, 262, "EngineCoreOutputs", color="#C53030",
                   lx=440, ly=252))
    s.append(arrow(850, 150, 910, 150, "SchedulerOutput", color="#C53030",
                   lx=880, ly=140))
    s.append(arrow(910, 262, 850, 262, "ModelRunnerOutput", color="#553C9A",
                   lx=880, ly=252))

    # 底部说明
    notes = [
        "输入通道：ZMQ ROUTER(bind) → DEALER(identity=engine_id)，msgspec msgpack",
        "输出通道：ZMQ PUSH(linger=4000) → PULL，按 client 分帧",
        "EngineCore 内部：两个 IO daemon 线程与主 busy loop 之间只经 queue.Queue",
        "EngineCore → Worker：multiprocessing.Process + MessageQueue 共享内存广播；单卡走同进程直调",
        "张量跨进程：tensor IPC 零拷贝 + zmq.MessageTracker 保引用",
    ]
    for i, n in enumerate(notes):
        s.append(f'<text x="{40 if i % 2 == 0 else 700}" y="{712 + (i // 2) * 24}" '
                 f'font-size="12.5" fill="#4A5568">· {esc(n)}</text>')

    s.append(svg_close())
    write("process-topology.svg", "\n".join(s))


# --------------------------------------------------------------------------
# 图 3：一次生成请求的端到端流程（泳道）
# --------------------------------------------------------------------------
def diagram_request_flow():
    W, H = 1360, 1000
    lanes = [
        ("客户端", "client"),
        ("API Server（协议 / 渲染 / 组包）", "entry"),
        ("EngineCore（排队 / 调度）", "core"),
        ("Worker（GPU 执行）", "exec"),
    ]
    lane_x = [40, 300, 660, 1020]
    lane_w = [240, 340, 340, 300]

    s = [svg_open(W, H, "一次生成请求的端到端流程（/v1/chat/completions）")]

    for (name, kind), x, w in zip(lanes, lane_x, lane_w):
        fill, stroke = PALETTE[kind]
        s.append(
            f'<rect x="{x}" y="76" width="{w}" height="700" rx="12" fill="{fill}" '
            f'fill-opacity="0.35" stroke="{stroke}" stroke-width="1.4"/>'
            f'<text x="{x + w / 2:.0f}" y="96" text-anchor="middle" font-size="14.5" '
            f'font-weight="700" fill="{stroke}">{esc(name)}</text>'
        )

    steps = [
        # (lane, y, [lines])
        (0, 112, ["POST /v1/chat/completions", "stream=true"]),
        (1, 112, ["validate_json_request", "OpenAIServingChat.create_chat_completion"]),
        (1, 176, ["Renderer.render_chat_async", "chat 模板 + 分词 + 多模态预处理"]),
        (1, 240, ["get_max_tokens → to_sampling_params"]),
        (1, 304, ["engine_client.generate（async generator）"]),
        (1, 368, ["InputProcessor.process_inputs", "→ EngineCoreRequest"]),
        (2, 368, ["Scheduler.add_request", "EngineCore 内排队 → WAITING"]),
        (2, 432, ["schedule()：allocate_slots / 抢占 / 前缀缓存", "→ SchedulerOutput"]),
        (3, 432, ["模型前向 + 采样", "→ ModelRunnerOutput"]),
        (2, 496, ["update_from_output + check_stop", "→ EngineCoreOutputs"]),
        (1, 496, ["OutputProcessor.process_outputs", "Detokenizer 增量反分词"]),
        (1, 560, ["RequestOutput 入队 → generate() 取出"]),
        (0, 560, ["SSE: data: {...chunk}"]),
        (1, 624, ["finish_reason 到达 → data: [DONE]"]),
        (0, 624, ["连接关闭"]),
    ]
    for lane, y, lines in steps:
        x = lane_x[lane]
        w = lane_w[lane]
        s.append(box(x + 16, y, w - 32, 56 if len(lines) > 1 else 44, lines,
                     lanes[lane][1], fs=12.5, lh=15))

    # 竖向时间箭头
    for x in (lane_x[0] + 40, lane_x[1] + 40, lane_x[2] + 40, lane_x[3] + 40):
        s.append(f'<line x1="{x}" y1="128" x2="{x}" y2="760" stroke="{PALETTE["support"][1]}" '
                 f'stroke-width="1" stroke-dasharray="4 6" opacity="0.45"/>')

    # 跨泳道箭头
    cross = [
        (0, 1, 140), (1, 2, 396), (2, 3, 460), (3, 2, 524),
        (2, 1, 524), (1, 0, 588), (1, 0, 652),
    ]
    for a, b, y in cross:
        if lane_x[a] < lane_x[b]:
            x1, x2 = lane_x[a] + lane_w[a] - 16, lane_x[b] + 16
        else:
            x1, x2 = lane_x[a] + 16, lane_x[b] + lane_w[b] - 16
        s.append(arrow(x1, y + 22, x2, y + 22, color="#4A5568"))

    # abort 分支
    s.append(box(1000, 792, 330, 104,
                 ["断连 / abort 路径", "with_cancellation 取消 handler",
                  "AsyncLLM.abort → OutputProcessor 清理",
                  "Scheduler.finish_requests（释放 KV）"], "kernel", fs=12, lh=17))
    s.append(arrow(140, 786, 1000, 800, "客户端断连 / POST /abort_requests",
                   color="#B83280", lx=560, ly=792))

    s.append(f'<text x="40" y="948" font-size="12" fill="#718096">'
             f'竖轴为时间顺序；横向箭头表示跨越进程/泳道的调用或数据流。</text>')
    s.append(svg_close())
    write("request-flow.svg", "\n".join(s))


# --------------------------------------------------------------------------
# 图 4：KV cache 三层结构 + 相关子系统
# --------------------------------------------------------------------------
def diagram_kv_stack():
    W, H = 1200, 720
    s = [svg_open(W, H, "KV cache 三层结构、offload 与显存账本")]

    s.append(band(30, 60, 700, 300, "逻辑分配（EngineCore）", "#C53030"))
    s.append(box(60, 96, 640, 60, ["KVCacheManager（门面，跨 KV group）",
                                   "allocate_slots:236 / free:437"], "core"))
    s.append(box(60, 172, 640, 60, ["KVCacheCoordinator（多 group 协调）",
                                    "full attention / sliding window / mamba"], "core"))
    s.append(box(60, 248, 640, 60, ["SingleTypeKVCacheManager（单类型，block 粒度）",
                                    "get_num_blocks_to_allocate:377"], "core"))
    s.append(arrow(380, 156, 380, 172, color="#C53030"))
    s.append(arrow(380, 232, 380, 248, color="#C53030"))

    s.append(band(30, 378, 700, 150, "物理块池与 GPU 张量", "#B7791F"))
    s.append(box(60, 414, 300, 96, ["BlockPool", "get_new_blocks:322 / free_blocks:408",
                                    "前缀缓存命中 + 引用计数"], "store"))
    s.append(box(390, 414, 310, 96, ["Worker 侧 KV 张量", "torch.zeros(int8, device)",
                                     "布局由 attention 后端决定"], "exec"))
    s.append(arrow(380, 308, 210, 414, "块 id", color="#B7791F", lx=250, ly=370))
    s.append(arrow(210, 414, 380, 308, "", color="#B7791F"))

    s.append(band(30, 546, 700, 142, "显存账本（启动期推算）", "#2F855A"))
    ledger = [
        "total_memory（设备总显存）",
        "× --gpu-memory-utilization",
        "− profile_run() 实测峰值",
        "− CUDA Graph 显存估算",
        "= 可用 KV 显存字节数",
        "÷ page_size ÷ num_layers",
        "= num_gpu_blocks",
    ]
    for i, t in enumerate(ledger):
        col, row = i % 2, i // 2
        s.append(box(50 + col * 350, 580 + row * 28, 336, 24, [t], "entry",
                     fs=11, lh=14, radius=5))

    s.append(band(760, 60, 410, 628, "卸载与缓存（三套独立机制）", "#553C9A"))
    off = [
        ["模型权重 offload", "config/offload.py：uva / prefetch", "model_executor/offloader/"],
        ["KV offload（分页式）", "v1/kv_offload/：OffloadKey + LRU/ARC", "tiering：CPU 为 primary 层"],
        ["KV offload（simple）", "v1/simple_kv_offload/：DmaCopyBackend", "cuMemcpyBatchAsync"],
        ["sleep mode", "device_allocator/cumem.py", "CuMemAllocator：weights / kv_cache pool"],
        ["前缀缓存（GPU 内）", "BlockPool cached blocks", "块哈希命中即跳过 prefill"],
        ["多模态 encoder 缓存", "v1/core/encoder_cache_manager.py", "物理张量在 gpu/mm/encoder_cache.py"],
    ]
    for i, lines in enumerate(off):
        s.append(box(780, 96 + i * 96, 370, 78, lines, "exec", fs=12, lh=16))

    s.append(svg_close())
    write("kv-cache-stack.svg", "\n".join(s))


# --------------------------------------------------------------------------
# 图 5：并行维度：每个维度切什么
# --------------------------------------------------------------------------
def diagram_parallelism():
    W, H = 1200, 700
    s = [svg_open(W, H, "并行维度：每个维度切分什么")]

    rows = [
        ("TP  张量并行", "权重矩阵按行/列切 + hidden 维 sequence parallel",
         "Column / Row / QKV ParallelLinear", "engine"),
        ("PP  流水并行", "按层切（layer stage），其余层用 PPMissingLayer 占位",
         "make_layers:620 与 get_pp_indices", "engine"),
        ("DP  数据并行", "切请求与批次；每 rank 一份完整权重",
         "DPAsyncMPClient / DPLBAsyncMPClient", "entry"),
        ("EP  专家并行", "切 MoE 专家（仅 MoE 模型建组）；= DP×PCP×TP",
         "FusedMoE(tp_size, ep_size) / ExpertMapManager", "core"),
        ("EPLB", "专家副本与重排（冗余专家）",
         "EplbState:210 + DefaultEplbPolicy", "store"),
        ("PCP  prefill 上下文", "prefill 阶段的序列/上下文切分",
         "get_pcp_group:1616", "model"),
        ("DCP  decode 上下文", "decode 阶段的 KV head 切分（dcp_size ≤ tp_size）",
         "get_dcp_group:1594", "model"),
    ]
    for i, (name, what, where, kind) in enumerate(rows):
        y = 84 + i * 84
        s.append(box(40, y, 250, 68, [name], kind, fs=14))
        s.append(box(300, y, 470, 68, [what], "support", fs=12.5))
        s.append(box(780, y, 380, 68, [where], "support", fs=11.5))

    s.append(f'<text x="40" y="672" font-size="12.5" fill="#4A5568">'
             f'组布局顺序：ExternalDP × DP × PP × PCP × TP；本版本没有独立的 CP 组，'
             f'context parallel 拆成 PCP 与 DCP。</text>')
    s.append(svg_close())
    write("parallelism.svg", "\n".join(s))


def linear_flow(name, title, steps, per_row=4, box_w=272, box_h=78, gap=24,
                kind="engine", note=None):
    """Render a left-to-right, wrapping flow of boxes with connecting arrows.

    steps: list of (main, sub) or (main, sub, kind).
    """
    rows = (len(steps) + per_row - 1) // per_row
    W = 80 + per_row * box_w + (per_row - 1) * gap
    H = 120 + rows * box_h + (rows - 1) * 74 + (64 if note else 40)
    s = [svg_open(W, H, title)]

    pos = []
    for i, step in enumerate(steps):
        r, c = divmod(i, per_row)
        x = 40 + c * (box_w + gap)
        y = 92 + r * (box_h + 74)
        pos.append((x, y, box_w, box_h))
        main, sub = step[0], step[1]
        k = step[2] if len(step) > 2 else kind
        s.append(box(x, y, box_w, box_h, [main, sub], k, fs=12.5, lh=16))

    for i in range(len(steps) - 1):
        x1, y1, w1, h1 = pos[i]
        x2, y2, w2, h2 = pos[i + 1]
        if y1 == y2:  # same row: horizontal arrow
            s.append(arrow(x1 + w1, y1 + h1 / 2, x2, y2 + h2 / 2, color="#4A5568"))
        else:  # wrap: elbow down-left
            midy = y1 + h1 + 37
            cx1, cx2 = x1 + w1 / 2, x2 + w2 / 2
            s.append(
                f'<polyline points="{cx1:.0f},{y1 + h1} {cx1:.0f},{midy} '
                f'{cx2:.0f},{midy} {cx2:.0f},{y2}" fill="none" stroke="#4A5568" '
                f'stroke-width="1.8" marker-end="url(#arrow)"/>'
            )

    if note:
        s.append(f'<text x="40" y="{H - 26}" font-size="12.5" fill="#4A5568">'
                 f'{esc(note)}</text>')
    s.append(svg_close())
    write(name, "\n".join(s))


# --------------------------------------------------------------------------
# 图 6：执行器拓扑
# --------------------------------------------------------------------------
def diagram_executors():
    W, H = 1240, 470
    s = [svg_open(W, H, "执行器拓扑与默认选择规则")]

    s.append(band(30, 66, 380, 330, "uni：world_size == 1", "#553C9A"))
    s.append(box(56, 106, 328, 74, ["UniProcExecutor", "同进程直接调用"], "exec"))
    s.append(arrow(220, 180, 220, 210, color="#553C9A"))
    s.append(box(56, 212, 328, 74, ["Worker", "持有 CUDA context 与权重"], "exec"))
    s.append(arrow(220, 286, 220, 316, color="#553C9A"))
    s.append(box(56, 318, 328, 60, ["GPUModelRunner"], "exec"))

    s.append(band(430, 66, 380, 330, "mp：多卡同机", "#C53030"))
    s.append(box(456, 106, 328, 74, ["MultiprocExecutor", "collective_rpc"], "core"))
    s.append(arrow(620, 180, 620, 210, "MessageQueue 广播", color="#C53030", lx=620, ly=199))
    s.append(box(456, 212, 328, 74, ["WorkerProc 0..N-1", "每个 rank 独立进程"], "core"))
    s.append(arrow(620, 286, 620, 316, color="#C53030"))
    s.append(box(456, 318, 328, 60, ["各自 CUDA context / 权重分片 / KV"], "core"))

    s.append(band(830, 66, 380, 330, "ray：跨节点", "#2F855A"))
    s.append(box(856, 106, 328, 74, ["RayExecutorV2", "Ray Actor 即 Worker"], "entry"))
    s.append(arrow(1020, 180, 1020, 210, color="#2F855A"))
    s.append(box(856, 212, 328, 74, ["Ray placement group", "跨节点 GPU 编排"], "entry"))
    s.append(arrow(1020, 286, 1020, 316, color="#2F855A"))
    s.append(box(856, 318, 328, 60, ["NCCL / IB 通信"], "entry"))

    s.append(f'<text x="40" y="428" font-size="12.5" fill="#4A5568">'
             f'默认选择（config/parallel.py:831-876）：CUDA 且多节点强制 mp；'
             f'本机 GPU 数不足直接报错；world_size 为 1 时用 uni。</text>')
    s.append(svg_close())
    write("executor-topology.svg", "\n".join(s))


# --------------------------------------------------------------------------
# 图 7：Worker 生命周期
# --------------------------------------------------------------------------
def diagram_worker_lifecycle():
    steps = [
        ("Worker.__init__", "精度设置 / elastic EP / weight transfer"),
        ("init_device", "设备与 dtype 校验，先建 NCCL 再取显存快照"),
        ("load_model", "CuMem weights pool + allocator 调参"),
        ("determine_available_memory", "profile_run + CUDA Graph 显存估算"),
        ("initialize_from_config", "分配 KV 张量，初始化 KV connector"),
        ("compile_or_warm_up_model", "kernel warmup，捕获 CUDA Graph"),
        ("execute_model / sample_tokens", "循环执行：前向与采样"),
        ("sleep / wake_up / shutdown", "CuMem pool 换入换出与退出"),
    ]
    linear_flow("worker-lifecycle.svg", "Worker 生命周期（gpu_worker.py）", steps,
                per_row=4, kind="exec",
                note="调用点：v1/worker/gpu_worker.py:107 / :239 / :338 / :354 / "
                     ":539 / :574 / :783 / :160")


# --------------------------------------------------------------------------
# 图 8：输入侧总管线
# --------------------------------------------------------------------------
def diagram_input_pipeline():
    steps = [
        ("HTTP chat 请求", "messages 含文本与图片/音频/视频"),
        ("chat_utils", "解析 content part，下载媒体，注入占位符"),
        ("Renderer.render_chat_async", "chat 模板 + 分词（renderers/base.py:998）"),
        ("_process_multimodal", "调 HF processor 产出 mm_kwargs"),
        ("BaseMultiModalProcessor.apply", "processor.py:1663 计算 mm_hashes"),
        ("MultiModalInput", "mm_kwargs / mm_hashes / mm_placeholders"),
        ("InputProcessor.process_inputs", "展平为 mm_features 列表"),
        ("EngineCoreRequest", "进入调度与 encoder 缓存"),
    ]
    linear_flow("input-pipeline.svg", "输入侧总管线：从 HTTP 请求到 EngineInput", steps,
                per_row=4, kind="entry",
                note="相关缓存：处理器缓存在 API Server 进程（CPU/shm），"
                     "encoder 缓存在 Worker（GPU 张量）")


# --------------------------------------------------------------------------
# 图 9：attention 数据流
# --------------------------------------------------------------------------
def diagram_attention_flow():
    steps = [
        ("Scheduler.schedule()", "产出 SchedulerOutput.block_ids"),
        ("GPUModelRunner.prepare_attn", "gather_block_tables + slot_mapping"),
        ("BlockTables", "device 上的 block table 张量"),
        ("AttentionMetadataBuilder.build", "后端专属 metadata"),
        ("AttentionImpl.forward", "paged / varlen kernel"),
        ("KV cache 张量", "由 initialize_kv_cache 分配"),
        ("hidden states", "交给 sampler 采样"),
        ("ModelRunnerOutput", "回传 EngineCore"),
    ]
    linear_flow("attention-dataflow.svg", "一次 attention 的数据流", steps,
                per_row=4, kind="model",
                note="后端优先级表在 platforms/cuda.py:78-143；"
                     "KV 布局由后端 get_kv_cache_shape / stride_order 决定")


# --------------------------------------------------------------------------
# 图 10：结构化输出 bitmask 链路
# --------------------------------------------------------------------------
def diagram_bitmask_flow():
    steps = [
        ("请求 structured_outputs", "json / regex / choice / grammar"),
        ("SamplingParams._validate", "sampling_params.py:787 选定后端"),
        ("StructuredOutputManager.grammar_init", "首次构造唯一 backend"),
        ("backend.compile_grammar", "编译 FSM 或语法"),
        ("Scheduler.get_grammar_bitmask", "scheduler.py:1222 组装 GrammarOutput"),
        ("manager.grammar_bitmask", "逐请求 fill_bitmask"),
        ("apply_grammar_bitmask", "structured_output/utils.py:44 重排并 H2D"),
        ("logits 原地 mask 到采样", "非法 token 置为 -inf"),
    ]
    linear_flow("bitmask-flow.svg", "结构化输出：从请求到 logits 掩码", steps,
                per_row=4, kind="core",
                note="后端选择：auto 先试 xgrammar，失败退 guidance / outlines；"
                     "FSM 推进在 scheduler.py:1357")


# --------------------------------------------------------------------------
# 图 11：指标通路
# --------------------------------------------------------------------------
def diagram_metrics_path():
    steps = [
        ("Scheduler.make_stats", "scheduler.py:1863 得到 SchedulerStats"),
        ("EngineCoreOutputs", "core.py:1808 放入 stats 字段"),
        ("AsyncLLM.output_handler", "async_llm.py:656 建 IterationStats"),
        ("OutputProcessor.process_outputs", "填充 iteration 与 FinishedRequest"),
        ("StatLoggerManager.record", "async_llm.py:697 分发"),
        ("LoggingStatLogger", "周期文本日志"),
        ("PrometheusStatLogger", "gauge / counter / histogram"),
        ("GET /metrics", "entrypoints 挂载的 Prometheus app"),
    ]
    linear_flow("metrics-path.svg", "指标通路：从 scheduler 到 /metrics", steps,
                per_row=4, kind="support",
                note="另有 get_metrics_snapshot（reader.py:70）供 LLM.get_metrics() 使用")


if __name__ == "__main__":
    diagram_overview()
    check_overlaps()
    diagram_process()
    diagram_request_flow()
    diagram_kv_stack()
    diagram_parallelism()
    diagram_executors()
    diagram_worker_lifecycle()
    diagram_input_pipeline()
    diagram_attention_flow()
    diagram_bitmask_flow()
    diagram_metrics_path()
    if WARNINGS:
        print(f"\n{len(WARNINGS)} layout warning(s):")
        for w in WARNINGS:
            print("  -", w)
    else:
        print("\nlayout check: no overflow or overlap detected")
