"""⑨ 三套真实配方：为什么它这么配。"""
from common import head, concl, anchors, legend, bridge, panel
from dg import Doc, table

TITLE = "⑨ 五套真实部署包：逐个参数对上「为什么」"
SUB = ("一句话：同一个模型，本仓有五种配法——差别几乎全部来自"
       "<b>「一卡放不放得下整模型」</b>这一件事。")


def build(doc: Doc) -> None:
    p = doc.page("⑨ 五套配方", pw=1560, ph=1560)
    head(p, TITLE, SUB)
    X = 36

    y = 148
    p.text(X, y - 32, 1300, 22,
           "<b>第一步：五套配方总表（数值全部来自脚本实测）</b>", "text_b")
    rows = [
        ["sidecar / in_process", "2 / 1", "65536", "128", "0.90", "3",
         "D: FULL_DECODE_ONLY", "P/edge: eager"],
        ["dp2_tp1_194_195", "1 / 2", "32768", "32", "0.95", "3",
         "D: FULL_DECODE_ONLY", "P/edge: eager"],
        ["mooncake_pd2_tp1_195", "1 / 2", "<b>512</b>", "32", "<b>0.98</b>", "3",
         "D: FULL_DECODE_ONLY", "—"],
        ["mooncake_pd_195", "2 / 1", "65536", "128", "0.95", "3",
         "D: FULL_DECODE_ONLY + 手写 25 档桶", "P: eager"],
        ["fix_1120", "<b>8</b> / 1", "65536", "128", "0.90", "<b>1</b>",
         "D: FULL_DECODE_ONLY", "P: eager + EP + 量化 + block 128"],
    ]
    table(p, X, y, [("包", 300), ("TP/DP", 140), ("max-model-len", 200),
                    ("max-num-seqs", 190), ("gpu-mem", 150), ("K", 90),
                    ("图（D 侧）", 250), ("其它", 264)], rows, row_h=44)

    # --- 逐条解释 ---------------------------------------------------------
    y2 = y + 44 * 6 + 40
    p.text(X, y2 - 32, 1300, 22,
           "<b>第二步：四条「为什么」（这才是配方的价值）</b>", "text_b")
    p.box(X, y2, 1424, 116,
          "<font style=\"font-size:10px\">"
          "<b>① 为什么 TP=2 的包敢报 65536，而 TP=1 的包只敢报 32768 甚至 512</b>："
          "TP=2 把权重切成两半，单卡剩余显存多 → 能装长上下文的 KV；"
          "TP=1 要在一张卡里放下整模型，剩余只够短上下文。<br>"
          "<b>② 为什么 TP=1 + MTP 的包把 max_num_seqs 压到 32</b>："
          "MTP 又多占一份 draft 开销与 lookahead 槽，KV 只剩几十个块，一个序列一块 → 32 是上限。<br>"
          "<b>③ 为什么 gpu-memory-utilization 有 0.90 / 0.95 / 0.98 三档</b>："
          "LWD 边云要留主机面与图池余量 → 0.90；普通 PD → 0.95；"
          "一卡塞整模型且还想开 MTP → 只能拉到 0.98 挤出来。<br>"
          "<b>④ 为什么只有 fix_1120 显式给 block-size 128 + 量化 + 专家并行</b>："
          "DeepSeek 那套要按组对齐块大小、要 8 卡放专家、要 int8 压 KV —— 三者是同一个目的。</font>",
          "box", "neutral")

    # --- 两种失败模式 -----------------------------------------------------
    y3 = y2 + 156
    p.text(X, y3 - 32, 1300, 22,
           "<b>第三步：抄配方时最容易犯的两个错</b>", "text_b")
    panel(p, X, y3, 700, 160, "错法一：只抄 K，不抄桶", [
        "K=3 意味着 decode 的 query 长度是 4，图桶必须是 4 的倍数。",
        "抄了 K 却把自己以前的桶列表一起抄过来（那些是 K=0 时代的数）→",
        "桶被自动取整或直接报错；更糟的是运行时静默回 eager。",
    ], "fail")
    panel(p, X + 760, y3, 664, 160, "错法二：只抄 max_model_len，不抄 seqs/gpu-mem", [
        "这三者是一条链：mml 大 → KV 需求大 → seqs 必须小，或 gpu-mem 必须高。",
        "只把 mml 抄大而 seqs、gpu-mem 不动 → 启动时报「KV 不够」并给出可支持的估计值。",
    ], "fail")

    y4 = y3 + 200
    p.box(X, y4, 1424, 96,
          "<b>另外两个实测发现（值得单独记）</b><br>"
          "<font style=\"font-size:10px\">"
          "① 有两套包（sidecar 与 in_process）<b>四个脚本逐字节相同</b>——它们是同一个配置的两种目录布局，"
          "画图/对比时应合并成一个节点。<br>"
          "② 有 6 个脚本是 <b>0 字节空壳</b>（rdma/tcp 下的三件套），目录名里的规模参数"
          "不代表任何生效配置——别从目录名推参数。</font>",
          "box", "data")

    y5 = y4 + 136
    bridge(p, y5, "本页交接：配方看完了。下一步（⑩）给一张一页纸速查表 + 五道自测题。")

    y6 = y5 + 76
    anchors(p, y6, [
        "sidecar：assets/c_home/bash/pds/30rc_1124/sidecar/cloud_p.sh:22-31、cloud_d.sh:22-31",
        "dp2：assets/c_home/bash/pds/30rc_1124/dp2_tp1_194_195/cloud_p.sh:33-41",
        "m2（mml 512 / gpu 0.98）：assets/c_home/bash/pds/30rc_1124/mooncake_pd2_tp1_195/env.sh:61-64",
        "m1（手写桶）：assets/c_home/bash/pds/30rc_1124/mooncake_pd_195/decode.sh:34",
        "f12（TP=8/EP/量化/block 128）：assets/c_home/bash/pds/fix_1120/prefill.sh:102-116",
        "桶必须按 K+1 对齐：vllm/config/compilation.py:1547-1592",
        "mml 放不下的报错：vllm/v1/core/kv_cache_utils.py:872-909",
    ], h=128)

    y7 = y6 + 162
    concl(p, y7,
          "<b>一句话记住这一页：</b>五套配方的差异归根到底是「一卡放不放得下」："
          "放不下就加 TP、压上下文、压并发；放得下就敢开长上下文与高并发。"
          "<b>抄参数要连带抄它的三个邻居（K 与桶、mml 与 seqs、gpu-mem）。</b>")

    legend(p, y7 + 92)
