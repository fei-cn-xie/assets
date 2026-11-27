"""② 参数地图：每条启动参数落在哪一格稀缺资源上。"""
from common import head, concl, anchors, legend, bridge
from dg import Doc, table

TITLE = "② 参数地图：本仓 13 个 launcher 里，哪些参数真正被改过"
SUB = ("一句话：vLLM 有上百个参数，但本仓部署实际只动了<b>十来个</b>；"
       "而且改哪几个完全由「硬件规模 + 要不要 MTP + 要不要图」决定。"
       "先看清这些，再谈联动。")


def build(doc: Doc) -> None:
    p = doc.page("② 参数地图", pw=1560, ph=1560)
    head(p, TITLE, SUB)
    X = 36

    y = 150
    p.text(X, y - 32, 1300, 22,
           "<b>第一步：六类参数（本仓实际出现的取值）</b>", "text_b")
    rows = [
        ["并行", "tp / dp / pp", "1、2、8（dp 最多 2）", "决定进程数与每卡分到多少权重"],
        ["长度与批量", "max-model-len / max-num-seqs / max-num-batched-tokens",
         "512~65536 / 32 或 128 / 4096 或 8192", "直接决定 KV 需求与并发上限"],
        ["显存水位", "gpu-memory-utilization", "0.90 / 0.95 / 0.98",
         "越高越能挤 KV，但留给图池与碎片的余量越少"],
        ["投机", "speculative-config", "mtp，K=3（有一个包 K=1）", "形状变 1+K、多占 lookahead 槽"],
        ["图", "enforce-eager / compilation-config",
         "P 与 edge: eager；D: FULL_DECODE_ONLY", "决定有没有图池与启动时间"],
        ["PD/KV", "kv-transfer-config / additional-config", "producer/consumer + kv_port",
         "决定 KV 往哪送、端口怎么派生"],
    ]
    table(p, X, y, [("类别", 180), ("参数", 420), ("本仓实际取值", 420),
                    ("它主要影响什么", 400)], rows, row_h=40)

    # --- 完全没改的 -------------------------------------------------------
    y2 = y + 40 * 7 + 40
    p.text(X, y2 - 32, 1300, 22,
           "<b>第二步：六个「13 个 launcher 里零命中」的参数——别以为它们被调过</b>",
           "text_b")
    p.box(X, y2, 1424, 108,
          "<font style=\"font-size:10px\">"
          "<b>--kv-cache-memory / --swap-space / --enable-chunked-prefill / "
          "--long-prefill-token-threshold / --scheduling-policy</b>：全部走默认；"
          "<b>--enable-expert-parallel</b> 只有 f12（TP=8 的 DeepSeek 包）显式开。<br>"
          "<font color=\"#c0392b\">这意味着：本仓的 chunked prefill 是默认开启的（当前版本默认 True），"
          "没有做任何阈值调整；显存分配也完全交给 gpu-memory-utilization 一个旋钮。</font></font>",
          "box", "fail")

    # --- 三个「同参数不同值」 ---------------------------------------------
    y3 = y2 + 148
    p.text(X, y3 - 32, 1300, 22,
           "<b>第三步：三个最典型的取舍（同一个参数，不同包取值差 100 倍）</b>",
           "text_b")
    p.box(X, y3, 460, 168,
          "<b>max-model-len：65536 / 32768 / <font color=\"#c0392b\">512</font></b><br>"
          "<font style=\"font-size:10px\">"
          "TP=2 的包敢报 65536；TP=1 + DP=2 的包退到 32768；"
          "而 TP=1 还要开 MTP 的包索性压到 <b>512</b>。<br>"
          "原因：单卡放完整模型后 KV 只剩零头，"
          "不压低上下文就起不来。</font>",
          "box", "data")
    p.box(X + 482, y3, 460, 168,
          "<b>max-num-seqs：128 vs 32</b><br>"
          "<font style=\"font-size:10px\">"
          "TP=2 的包可以 128；TP=1 + MTP 的包只能 32"
          "（MTP 把 KV 吃到只剩几十个块，一个序列一块）。<br>"
          "这是本仓注释里明说的：「64 会在启动时被拒」。</font>",
          "box", "fail")
    p.box(X + 964, y3, 460, 168,
          "<b>gpu-memory-utilization：0.90 / 0.95 / 0.98</b><br>"
          "<font style=\"font-size:10px\">"
          "LWD 边云包留 0.90（要给主机面与图池留余量）；"
          "PD 包用 0.95；DP=2 单卡塞整模型的包拉到 0.98。<br>"
          "越高越省事，但离 OOM 越近。</font>",
          "box", "cloudp")

    # --- 参数与资源的对应 ------------------------------------------------
    y4 = y3 + 208
    p.box(X, y4, 1424, 96,
          "<b>把参数映射到 ① 的四格资源</b><br>"
          "<font style=\"font-size:10px\">"
          "显存 ← gpu-memory-utilization / max-model-len / max-num-seqs / block-size / "
          "speculative-config / compilation-config；"
          "算力 ← max-num-batched-tokens / chunked prefill；"
          "带宽 ← tp（并行度越高通信越多）/ kv-transfer；"
          "形状自由度 ← chunked prefill × 投机 K × 图模式。</font>",
          "box", "neutral")

    y5 = y4 + 136
    bridge(p, y5, "本页交底：真正被改的只有十来个参数。下一步（③）从最硬的一条联动开始——"
                  "投机解码的 K 会直接改写「图捕获的桶」。")

    y6 = y5 + 76
    anchors(p, y6, [
        "并行与长度参数（本仓取值）：assets/c_home/bash/pds/30rc_1124/sidecar/cloud_p.sh:22-31、"
        "**cloud_d.sh:22-31**",
        "TP=1 + MTP 的压缩取值：assets/c_home/bash/pds/30rc_1124/dp2_tp1_194_195/cloud_p.sh:40-41、"
        "**mooncake_pd2_tp1_195/env.sh:61-64**",
        "TP=8 / EP / 量化 / block-size：assets/c_home/bash/pds/fix_1120/prefill.sh:102-116",
        "显式 cudagraph_capture_sizes 的包：assets/c_home/bash/pds/30rc_1124/mooncake_pd_195/decode.sh:34",
        "零命中的默认参数：vllm/config/scheduler.py:116（enable_chunked_prefill 默认 True）、"
        "vllm/config/cache.py:71（block_size 默认 16，本仓 f12 显式 128）",
        "全部 launcher 都开 async scheduling：各脚本的 --async-scheduling",
    ], h=120)

    y7 = y6 + 154
    concl(p, y7,
          "<b>一句话记住这一页：</b>本仓只改十来个参数，核心是「并行 × 长度/批量 × 显存水位 × "
          "是否 MTP × 是否图」；<b>同一参数在不同包差 100 倍，差别全来自「一卡放得下多少」这个物理约束。</b>")

    legend(p, y7 + 92)
