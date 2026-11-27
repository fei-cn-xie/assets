"""④ 联动二：显存的三条链（权重/图池/MTP 各自怎么吃掉 KV）。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc

TITLE = "④ 联动二：想给 KV 多留一点，先搞清楚是谁在吃它"
SUB = ("一句话：可用 KV = <b>上限 − 权重 − 激活峰值 − 图池 − 碎片</b>；"
       "所以「权重变大」「开图捕获」「开 MTP」都会让并发变小——只是路径不同。")


def build(doc: Doc) -> None:
    p = doc.page("④ 显存三条链", pw=1560, ph=1520)
    head(p, TITLE, SUB)
    X = 36

    # --- 公式 -------------------------------------------------------------
    y = 150
    p.box(X, y, 1424, 92,
          "<font style=\"font-size:11px\">"
          "<b>可用 KV = gpu_memory_utilization × 卡总显存 − 非 KV 部分</b><br>"
          "<font style=\"font-size:10px\">"
          "其中「非 KV 部分」= 权重 + 激活峰值（profile 量出）+ 图池估算（可选计入）+ 碎片；"
          "剩下的全部被切成块 → 块数 ÷ 每请求块数 = 能开多少并发。</font></font>",
          "box", "neutral")

    # --- 三条链 -----------------------------------------------------------
    y2 = y + 130
    p.text(X, y2 - 34, 1300, 22,
           "<b>第一步：三条「减少 KV」的路径（本仓都能找到实例）</b>", "text_b")
    panel(p, X, y2, 460, 200, "链一：权重变大 → KV 变少", [
        "TP=1 时整模型都压在一张卡上：",
        "权重占比越高 → 剩余越少 → 块数越少 → 并发越低。",
        "",
        "本仓表现：TP=1 + MTP 的包只能报 max_num_seqs=32，",
        "甚至把 max_model_len 压到 512 才起得来。",
        "（TP=2 的包就能报 65536 / 128）",
    ], "data")
    panel(p, X + 482, y2, 460, 200, "链二：开图捕获 → KV 变少", [
        "图池是一整块进程级内存（≈各桶峰值取 max），",
        "估算时会被从可用显存里扣掉（可开关）。",
        "",
        "本仓表现：P 与 edge 用 --enforce-eager 正是为了省这块；",
        "D 侧要 decode 吞吐，所以接受这块开销。",
    ], "cloudd")
    panel(p, X + 964, y2, 460, 200, "链三：开 MTP → KV 变少", [
        "两块开销：",
        "① draft 权重与缓冲（MTP 复用 embedding/lm_head，增量小）；",
        "② <b>每请求多留 K 个 lookahead 槽</b>（草案要往 target 之外写 KV）。",
        "",
        "本仓注释明说：MTP 把 KV 吃到只剩几十个块。",
    ], "fail")

    # --- 两个覆盖关系 -----------------------------------------------------
    y3 = y2 + 240
    p.text(X, y3 - 34, 1300, 22,
           "<b>第二步：两个「覆盖」关系（设了就生效，别指望叠加）</b>", "text_b")
    panel(p, X, y3, 700, 148, "kv_cache_memory_bytes 会无视 gpu_memory_utilization", [
        "一旦直接给 KV 分配字节数，引擎会跳过 memory profiling，",
        "完全按你给的数切块（日志里会明说「不遵守 gpu_memory_utilization」）。",
        "→ 想精确控制就给它；想省心就用百分比。二者只信一个。",
    ], "fail")
    panel(p, X + 760, y3, 664, 148, "max_model_len 放不下 KV 直接起不来", [
        "报错会告诉你「KV 需要多少、可用多少」，",
        "并给出二分搜索出来的「本卡最多能支持多长」。",
        "本仓有一个包正是靠这一条把 max_model_len 压到 512 才起得来。",
    ], "fail")

    # --- 互相拆台的组合 ---------------------------------------------------
    y4 = y3 + 188
    p.box(X, y4, 1424, 108,
          "<b>第三步：三个「同时开就互相拆台」的组合</b><br>"
          "<font style=\"font-size:10px\">"
          "① <b>TP=1 + MTP + 大 max_model_len</b>：权重 + MTP 开销已经吃满，KV 只剩零头 → 起不来。"
          "本仓解法是把 mml 压到 512 或干脆不开 MTP。<br>"
          "② <b>FULL_DECODE_ONLY + 小 max_num_seqs</b>：图池按最大桶抓，桶又由 seqs×(1+K)×2 决定 → "
          "seqs 太小会让桶太小、图收益变小。<br>"
          "③ <b>高 gpu_memory_utilization + 图捕获</b>：水位拉到 0.98 时，"
          "图池与碎片没有余量 → 捕获阶段容易 OOM（而捕获段没有回退）。</font>",
          "box", "fail")

    y5 = y4 + 148
    bridge(p, y5, "本页交接：显存的三条链讲完。下一步（⑤）转到调度参数——"
                  "那里有一整条「你没设，引擎替你算」的推导链。")

    y6 = y5 + 76
    anchors(p, y6, [
        "可用 KV 的计算：vllm/v1/worker/gpu_worker.py:619-625；"
        "图池估算开关 :588-592",
        "kv_cache_memory_bytes 覆盖：vllm/config/cache.py:228-235；"
        "vllm/v1/worker/gpu_worker.py:542-564",
        "max_model_len 放不下的报错与拟合：vllm/v1/core/kv_cache_utils.py:872-909、"
        ":921-972（auto fit）",
        "lookahead 槽：vllm/config/vllm.py:633-658；scheduler.py:744",
        "块数随 block_size 反比：vllm/v1/core/kv_cache_utils.py:1747-1751",
        "本仓取值：assets/c_home/bash/pds/30rc_1124/dp2_tp1_194_195/cloud_p.sh:40-41（"
        "seqs 32 / gpu 0.95）、mooncake_pd2_tp1_195/env.sh:61-64（mml 512 / gpu 0.98）",
    ], h=120)

    y7 = y6 + 154
    concl(p, y7,
          "<b>一句话记住这一页：</b>想让并发变大，只有三条路——"
          "减少权重占用（加 TP）、别开图池（eager）、别开 MTP；"
          "<b>而「提高 gpu_memory_utilization」只是把水位线抬高，代价是离 OOM 更近。</b>")

    legend(p, y7 + 92, extra_fill=("权重/显存压力", "data"))
