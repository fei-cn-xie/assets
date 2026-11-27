"""③ 联动一：投机解码的 K 改写图捕获的桶。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc, table

TITLE = "③ 联动一：开了 MTP，图捕获的「桶」必须整批重算"
SUB = ("一句话：投机解码把 decode 的 query 长度从 1 变成 1+K；"
       "而图捕获的桶是一份<b>硬编码的尺寸列表</b>——两者必须对上，"
       "对不上要么被自动取整、要么直接报错起不来。")


def build(doc: Doc) -> None:
    p = doc.page("③ K 改写桶", pw=1560, ph=1480)
    head(p, TITLE, SUB)
    X = 36

    # --- 因果链 -----------------------------------------------------------
    y = 152
    p.text(X, y - 34, 1300, 22,
           "<b>第一步：一条四步因果链（每一步都能在代码里指到）</b>", "text_b")
    chain = [
        ("①", "你设 K=3", "speculative-config 的 num_speculative_tokens", "cloudp"),
        ("②", "decode 的 query 长度\n变成 1+K = 4",
         "uniform_decode_query_len", "cloudp"),
        ("③", "图桶必须是 4 的倍数", "capture size = n×(K+1)", "cloudd"),
        ("④", "运行时断言\npadded % 4 == 0", "否则回不到这张图", "fail"),
    ]
    cw = 340
    for i, (no, title, note, role) in enumerate(chain):
        cx = X + i * (cw + 21)
        p.box(cx, y, cw, 118,
              f"<b>{no} {title.replace(chr(10), '<br>')}</b><br>"
              f"<font style=\"font-size:9px;color:#666666\">{note}</font>",
              "box_center", role)
        if i:
            p.edge(src_pt=(cx - 19, y + 59), dst_pt=(cx - 3, y + 59), st="arrow",
                   role="http")

    # --- 桶列表怎么变 -----------------------------------------------------
    y2 = y + 158
    p.text(X, y2 - 34, 1300, 22,
           "<b>第二步：同一组「最大 512」，开/不开 MTP 的桶不一样</b>", "text_b")
    panel(p, X, y2, 700, 176, "不开投机（K=0，query 长度 = 1）", [
        "桶可以是任意尺寸：1, 2, 4, 8, 16, 32, …, 512",
        "因为 query 长度恒为 1，padded token 数就等于请求数。",
        "所以「几个请求 → 落到哪个桶」很直观。",
    ], "neutral")
    panel(p, X + 760, y2, 664, 176, "开 MTP（K=3，query 长度 = 4）", [
        "桶必须是 4 的倍数：4, 8, 12, …, 512（实际由代码取整得到）",
        "因为一个请求就要占 4 个位置 → padded token 数 = 请求数 × 4。",
        "<font color=\"#c0392b\">于是「能同时跑几个请求」= 桶大小 ÷ 4——"
        "最大的桶 512 只能装 128 个请求。</font>",
    ], "cloudd")

    # --- 三种结果 ---------------------------------------------------------
    y3 = y2 + 216
    p.text(X, y3 - 34, 1300, 22,
           "<b>第三步：对不上会怎样（三种结果，都不太直观）</b>", "text_b")
    rows = [
        ["你显式给了 capture_sizes 且不是 K+1 的倍数",
         "引擎帮你向上取整到倍数", "启动会看到「尺寸被改写」的日志"],
        ["取整后超过 max_cudagraph_capture_size",
         "直接报错，起不来", "报错文案会点名「spec-decode 要求的 K+1」"],
        ["运行时批大小没对上倍数",
         "那一张图用不了 → 回落到 eager", "<b>没有日志</b>，只表现为变慢"],
    ]
    table(p, X, y3, [("情形", 620), ("结果", 420), ("你能看到什么", 380)], rows,
          row_h=40)

    # --- 本仓实例 ---------------------------------------------------------
    y4 = y3 + 40 * 4 + 36
    p.box(X, y4, 1424, 116,
          "<b>本仓的真实组合（可以直接对照）</b><br>"
          "<font style=\"font-size:10px\">"
          "· 绝大多数包：K=3 + <font style=\"font-size:9px\">FULL_DECODE_ONLY</font> → "
          "桶由引擎自动取整成 4 的倍数；<br>"
          "· 有一个包（mooncake_pd_195 的 decode）<b>手工写了 25 个 capture_sizes</b>"
          "（4, 8, …, 100）——这些数正好都是 4 的倍数，说明写它的人已经按 K+1 对齐过；<br>"
          "· 另一个包 K=1（DeepSeek-V4 只抽 1 个 draft）→ query 长度 2 → 桶是 2 的倍数。"
          "<font color=\"#c0392b\">所以「抄参数」时最容易错的就是只抄 K 不抄桶。</font></font>",
          "box", "data")

    y5 = y4 + 156
    bridge(p, y5, "本页交接：K 还牵动另外两样东西——每个请求的 token 预算、以及 KV 的 lookahead 槽。"
                  "下一步（④）把「显存的三条链」画出来：权重、图池、MTP 各自怎么吃掉 KV。")

    y6 = y5 + 76
    anchors(p, y6, [
        "K → query 长度：vllm/config/vllm.py:679；vllm/v1/worker/gpu_model_runner.py:905",
        "桶按 K+1 对齐（自动取整 + 报错文案）：vllm/config/compilation.py:1503-1518、"
        ":1547-1592",
        "运行时整除断言：vllm/v1/cudagraph_dispatcher.py:144-145",
        "昇腾侧同款要求：vllm-ascend/docs/.../speculative_decoding.md:155-158、:222；"
        "vllm-ascend/vllm_ascend/compilation/compiler_interface.py:73-74",
        "每个请求的 batched-token 预算：vllm/v1/core/sched/scheduler.py:820、:1327",
        "本仓手工 capture_sizes：assets/c_home/bash/pds/30rc_1124/mooncake_pd_195/decode.sh:34",
    ], h=120)

    y7 = y6 + 154
    concl(p, y7,
          "<b>一句话记住这一页：</b>K 不只影响「猜几个」——它把 decode 的形状改成 1+K，"
          "于是图桶、每请求 token 预算、KV lookahead 三样都要跟着改；"
          "<b>只改 K 不改桶，是最常见的配置错。</b>")

    legend(p, y7 + 92)
