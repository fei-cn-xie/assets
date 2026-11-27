"""⑩ 收口：为什么 MoE 才需要 wave 协调 + 术语表 + 排障清单。"""
from common import head, concl, anchors, legend
from dg import Doc, table

TITLE = "⑩ 收口：为什么「wave 协调」只有 MoE 才需要 + 术语 + 排障"
SUB = ("一句话：MoE 的 DP rank <b>不是彼此独立的</b>——每层都要跨 rank 搬 token，"
       "所以所有 rank 必须<b>同时走、同时停</b>；这个「一起走一起停」就是 wave。"
       "dense 的 DP rank 才是真正各干各的。")


def build(doc: Doc) -> None:
    p = doc.page("⑩ wave 协调与收口", pw=1560, ph=1620)
    head(p, TITLE, SUB)
    X = 36

    # --- wave -------------------------------------------------------------
    y = 144
    p.text(X, y - 30, 1400, 22,
           "<b>第一步：为什么 MoE 的 DP rank 必须「一起走、一起停」</b>", "text_b")
    p.box(X, y, 1424, 132,
          "<font style=\"font-size:10px\">"
          "<b>dense 的 DP</b>：每张卡各自跑各自的请求，互不通信 → 一个 rank 忙、另一个 rank 闲，完全没问题。<br>"
          "<b>MoE 的 DP</b>：每一层都要跨 rank 做集合通信（搬 token / 归约），"
          "而集合通信<b>要求所有 rank 同时参与</b> → 只要有一个 rank 停下来了，其他 rank 就会卡在通信里。<br>"
          "<font color=\"#c0392b\">所以 MoE 的所有 rank 必须「同一时刻要么都在跑、要么都在停」——"
          "这就是 wave（波次）。代码里甚至有一条注释大意是："
          "「空微批也要跑成 dummy 批，因为大家必须切得一样多」。</font></font>",
          "box", "fail")

    y2 = y + 172
    p.from_nothing if False else None
    p.box(X, y2, 1424, 108,
          "<b>第二步：代码里的三个「硬证据」</b><br>"
          "<font style=\"font-size:10px\">"
          "① 一行定死：<font style=\"font-size:9px\">enable_wave_coordination = is_moe</font>"
          "——只有 MoE 模型才启用波次协调；<br>"
          "② 一条断言：MoE 专用的引擎进程类型上写着「只应该用于 MoE 模型」；<br>"
          "③ 一条纪律：同步状态结构体的注释写着「每个字段在所有 rank 上必须相同，"
          "<b>永远不要往里加「只在本 rank 有意义」的值</b>」——否则各 rank 对「要不要通信」的判断会不一致 → 死锁。</font>",
          "box", "neutral")

    # --- 术语表 -----------------------------------------------------------
    y3 = y2 + 148
    p.text(X, y3 - 30, 1400, 22,
           "<b>第三步：术语表（看到这些词不再卡住）</b>", "text_b")
    rows = [
        ["expert（专家）", "MoE 层里并列的 N 个小 FFN，各自一套权重"],
        ["router / gate（门卫）", "一个「hidden → N」的线性层，给每个 token 打分"],
        ["top-k", "每个 token 只保留分数最高的 k 个专家，其余不算"],
        ["permute / unpermute", "按专家把 token 聚到一起（重排）/ 算完散射回原位"],
        ["grouped GEMM", "一次提交里为每个专家各跑一段矩阵乘（各段 M/N/K 不同）"],
        ["w13 / w2", "专家权重的两个名字：gate+up 融合 / down（跨模型统一命名）"],
        ["shared expert（共享专家）", "一个所有 token 必走、权重恒 1 的 dense 旁路"],
        ["EP（专家并行）", "把 N 个专家切到多张卡；开 EP 时 MoE 层的 TP 被关掉"],
        ["EPLB", "推理期统计负载 → 把热门专家复制副本、挪到空闲卡（不动进程结构）"],
        ["wave（波次）", "MoE 的 DP rank 必须一起走/一起停；一次「从跑到停」算一波"],
        ["balancedness", "均衡度 = 每专家平均 token ÷ 每专家最大 token，越接近 1 越好"],
        ["capacity factor / token drop", "训练期概念（给专家预留槽位、超了就丢）；<b>本仓推理路径不做截断</b>"],
    ]
    table(p, X, y3, [("术语", 340), ("一句大白话", 1084)], rows, row_h=30)

    # --- 排障 -------------------------------------------------------------
    y4 = y3 + 30 * 13 + 32
    p.text(X, y4 - 30, 1400, 22,
           "<b>第四步：MoE 排障清单（八条，都能在仓里找到原文）</b>", "text_b")
    rows2 = [
        ["开了 EP 但启动报「必须是 MoE 模型」", "EP 校验要求模型有专家", "换模型或别开 EP"],
        ["融合 kernel 报错 / 结果不对", "融合算子只支持特定量化组合与硬件", "回退分步路径"],
        ["all-to-all 路径报「不支持 bias」", "该实现的显式限制", "换通信实现"],
        ["共享专家形状/结果不一致", "EP 改写了 MoE 层 tp，但共享专家按真实 TP 布局", "按真实 TP 组判断"],
        ["EPLB 拒绝启动", "同步 EPLB 不支持 / 与弹性 EP 冲突 / 通信库不是 gloo·hixl", "改异步、去冲突"],
        ["量化 MoE + 动态 EPLB 报错", "部分量化方案不提供 per-expert 权重视图", "关 EPLB 或换量化"],
        ["专家负载严重不均", "真实流量分布偏斜（不是 bug）", "开 EPLB 或调放置"],
        ["显存不够（即使开了 EP）", "EP 只摊开不减少总量；专家占权重 85~90%", "量化、减冗余、降并发"],
    ]
    table(p, X, y4, [("现象", 460), ("原因", 620), ("处置", 344)], rows2, row_h=30)

    # --- 一句话收 ---------------------------------------------------------
    y5 = y4 + 30 * 9 + 32
    p.box(X, y5, 1424, 116,
          "<b>五句话总结这一套</b><br>"
          "<font style=\"font-size:10px\">"
          "① MoE = 很多专家 + 门卫 + 每 token 只走 top-k：<b>总参数按专家数涨，计算按 top-k 涨，显存按专家数涨</b>；<br>"
          "② 一层 MoE 的计算只有「一次小线性 + k 段 GEMM」，中间夹两次数据搬运（重排/散射）；<br>"
          "③ 推理要付三笔账：<b>显存</b>（全部专家驻留）、<b>通信</b>（搬 token）、<b>负载</b>（热门专家拖慢某卡）；<br>"
          "④ 三笔账的解法分别是：EP 切分（摊开）、通信实现选择（默认 all-gather，可换融合/按需）、EPLB 换位；<br>"
          "⑤ 而 MoE 的 DP rank 因为要通信，<b>必须 wave 协同</b>——这是它和 dense 最本质的结构差别。</font>",
          "box", "cloudd")

    y6 = y5 + 156
    anchors(p, y6, [
        "wave 只给 MoE：vllm/v1/engine/utils.py:1199（enable_wave_coordination=is_moe）；"
        "断言 vllm/v1/engine/core.py:2083-2085；非 MoE DP 语义 :1381-1386",
        "「空微批也要跑 dummy 批」原文：vllm/v1/worker/gpu/dp_utils.py:115-116；"
        "同步状态纪律 :22-27、:78（all_reduce）",
        "wave 的定义与触发：vllm/v1/engine/coordinator.py:33-52、:473-479；"
        "共识点 core.py:2339-2356",
        "专家权重命名 w13/w2：vllm/model_executor/layers/fused_moe/layer.py:148-153（原文）",
        "两块 scratch 复用：modular_kernel.py:834-871、:1117-1196",
        "容量因子/丢 token 属训练期概念、本仓推理不截断：token_dispatcher.py:616-620",
        "MoE 相关限制原文：moe_comm_method.py:221-227（finalize_routing 精度问题改用 unpermute）、"
        ":243-245（MC2 要求开 EP）；platform.py:996-1017（EPLB 限制）；"
        "routed_experts.py:127-131、:462-467（FUSED_MC2 与量化要求）",
    ], h=152)

    y7 = y6 + 186
    concl(p, y7,
          "<b>一句话记住这套图：</b>MoE 用「稀疏」换来了参数容量，"
          "代价是显存、通信、负载三笔账，以及「所有 rank 必须同步走停」这条结构约束；"
          "<b>理解了这三笔账和 wave，MoE 的所有配置项就都有了解释。</b>")

    legend(p, y7 + 92)
