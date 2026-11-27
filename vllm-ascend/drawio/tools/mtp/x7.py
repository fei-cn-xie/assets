"""⑦ 三条系统约束：形状变 1+K、显存多留槽、调度要求 uniform。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc

TITLE = "⑦ 投机解码踩到的三条系统约束（与前面几套图接上）"
SUB = ("一句话：decode 的 query 长度从 1 变成 1+K，"
       "<b>形状变了（图捕获要补桶）、显存变了（要多留 lookahead 槽）、调度变了（要求 uniform decode）</b>。")


def build(doc: Doc) -> None:
    p = doc.page("⑦ 三条系统约束", pw=1560, ph=1500)
    head(p, TITLE, SUB)
    X = 36

    # --- 约束一 -----------------------------------------------------------
    y = 152
    p.text(X, y - 34, 1300, 22,
           "<b>约束一：形状 —— decode 的 query 长度 = 1 + K</b>", "text_b")
    panel(p, X, y, 700, 176, "发生了什么", [
        "普通 decode：每个请求这一步只有 1 个 query 位置；",
        "投机 decode：每个请求有 1+K 个位置（K 个草案 + 1 个 new）。",
        "所以「一次前向的 token 数」= 请求数 × (1+K)。",
        "代码里这个名字叫 uniform_decode_query_len。",
    ], "cloudp")
    panel(p, X + 760, y, 664, 176, "连带后果：图捕获的桶必须按 (K+1) 的倍数", [
        "图捕获要求固定形状 → 桶列表必须能被 K+1 整除，否则运行时会报错；",
        "vLLM 会把 capture size 向上取整到 K+1 的倍数；",
        "运行时还有断言：padded num_tokens % (K+1) == 0。",
        "→ 详见「vLLM图捕获」套 ⑥（桶与 dispatch）。",
    ], "fail")

    # --- 约束二 -----------------------------------------------------------
    y2 = y + 216
    p.text(X, y2 - 34, 1300, 22,
           "<b>约束二：显存 —— draft 自己占一份，KV 还要多留</b>", "text_b")
    panel(p, X, y2, 700, 190, "两块额外开销", [
        "① <b>draft 权重</b>：MTP/EAGLE 复用 target 的 embedding 与 lm_head，增量小；"
        "独立小模型则要整份新权重。",
        "② <b>lookahead 槽</b>：草案会往 target 之外写 KV，分配时必须多预留 K 个槽。",
        "③ 另外还有与 1+K 成正比的缓冲区（例如 topk buffer 按 (1+K) 放大）。",
    ], "fail")
    panel(p, X + 760, y2, 664, 190, "本仓实测：开 MTP 后 KV 明显变少", [
        "在 TP=1 + MTP 的包里，KV 块只剩几十个（32~63 个 GDN/Mamba 块），",
        "于是 max_num_seqs 被压到 32；关掉 MTP 时可用的 KV 会明显回升。",
        "→ 这就是「投机解码不是免费的」的现实版：它拿并发换单请求速度。",
    ], "data")

    # --- 约束三 -----------------------------------------------------------
    y3 = y2 + 230
    p.text(X, y3 - 34, 1300, 22,
           "<b>约束三：调度 —— 为了保住整图，必须让批「整齐」</b>", "text_b")
    panel(p, X, y3, 1424, 148, "为什么调度器要把「长度为 1 的 decode 请求」也 pad 成 1+K", [
        "如果一批里有的请求是 1 个 query、有的是 1+K 个，形状就不 uniform → "
        "整图（FULL cudagraph）用不了，只能退回分段图或 eager。",
        "所以调度器在「本步要出 token」时，会把纯 decode 请求也补齐到 1+K（缺的位置填 -1 占位），"
        "宁可多算几个空位，也要保住整图。",
        "代价：padding 是白算的算力；但如果丢了整图，损失更大。这就是「uniform decode」这个词的来源。",
    ], "neutral")

    # --- 一张总表 ---------------------------------------------------------
    y4 = y3 + 188
    p.text(X, y4 - 34, 1300, 22,
           "<b>把三条约束与前面的图对接</b>", "text_b")
    from dg import table
    rows = [
        ["形状 1+K", "图捕获的桶按 K+1 对齐；运行时整除断言", "图捕获 ⑥"],
        ["显存：draft + lookahead", "可用 KV 变少 → 并发下降；需按实测调 max_num_seqs", "入门套 ⑦ & 本套 ⑥"],
        ["调度：uniform decode", "scheduler 把纯 decode pad 成 1+K", "进程/线程 ⑫"],
        ["MTP 层独立 KV", "PD 传输时 MTP 层的 KV 作为额外层单独编号（只传一次）", "Mooncake ③ & 本套 ⑨"],
    ]
    table(p, X, y4, [("约束", 320), ("在代码/部署里的表现", 700), ("去哪看细节", 380)],
          rows, row_h=36)

    y5 = y4 + 36 * 5 + 36
    bridge(p, y5, "本页交接：这三条约束在昇腾上有更硬的上限（例如 TND 布局要求 K+1 ≤ 16）。"
                  "下一步（⑧）专门讲昇腾侧的落地。")

    y6 = y5 + 76
    anchors(p, y6, [
        "1+K 的定义：vllm/config/vllm.py:679；vllm/v1/worker/gpu_model_runner.py:905",
        "图桶按 K+1 对齐：vllm/config/compilation.py:1503-1518、:1547-1592；"
        "运行时断言 vllm/v1/cudagraph_dispatcher.py:144-145",
        "lookahead 预留：vllm/config/vllm.py:633-658；scheduler.py:741-745",
        "纯 decode pad 成 1+K：vllm/v1/core/sched/scheduler.py:1063-1073、:1330-1334",
        "MTP 层 KV 单独编号（PD 传输只传一次）："
        "vllm-ascend/.../kv_p2p/mooncake_connector.py:2391-2403",
        "本仓 KV 变少的部署注释：assets/c_home/bash/pds/30rc_1124/dp2_tp1_194_195/cloud_p.sh:12-16",
    ], h=120)

    y7 = y6 + 154
    concl(p, y7,
          "<b>一句话记住这一页：</b>投机解码不是「只在模型里加个头」——"
          "它把 decode 的形状改成 1+K、把显存切走一块给 draft 与 lookahead、"
          "还要求调度器把批次凑整齐；<b>这三条是它与图捕获/显存账/连续批处理三套图的接口</b>。")

    legend(p, y7 + 92, extra_fill=("显存/数据面相关", "data"))
