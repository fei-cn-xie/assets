"""⑤ 联动三：调度参数有一条「你没设、引擎替你算」的推导链。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc

TITLE = "⑤ 联动三：max_num_batched_tokens 不是你设的，是引擎算出来的"
SUB = ("一句话：如果你不设它，引擎会按<b>显存大小 → 用途 → 性能模式 → 是否分块 → 最大长度</b>"
       "依次推导；而它又会反过来决定图桶、每步能装多少 token、甚至 max_num_seqs。")


def build(doc: Doc) -> None:
    p = doc.page("⑤ 调度推导链", pw=1560, ph=1560)
    head(p, TITLE, SUB)
    X = 36

    # --- 推导链 -----------------------------------------------------------
    y = 150
    p.text(X, y - 32, 1300, 22,
           "<b>第一步：七步推导（只看「用户没显式设」的情形）</b>", "text_b")
    steps = [
        ("①", "按显存选默认", "大显存 16384 / 中等 8192 / 小 2048", "neutral"),
        ("②", "按用途选默认", "纯 LLM 与 API server 取值不同", "neutral"),
        ("③", "性能模式翻倍", "throughput 模式下默认值 ×2", "cloudp"),
        ("④", "关分块则抬高下限", "不分块时一步要能装下整条序列", "fail"),
        ("⑤", "多模态再抬", "至少能装下一个样本的 token", "neutral"),
        ("⑥", "按上限截断", "不超过 seqs × max_model_len", "cloudd"),
        ("⑦", "反推 max_num_seqs", "取默认值与 batched 的较小者", "cloudd"),
    ]
    cw = 196
    for i, (no, title, note, role) in enumerate(steps):
        cx = X + i * (cw + 8)
        p.box(cx, y, cw, 104,
              f"<b>{no} {title}</b><br>"
              f"<font style=\"font-size:9px;color:#666666\">{note}</font>",
              "box_center", role)
        if i:
            p.edge(src_pt=(cx - 7, y + 52), dst_pt=(cx - 1, y + 52), st="arrow",
                   role="http")

    # --- 四条硬校验 -------------------------------------------------------
    y2 = y + 144
    p.text(X, y2 - 32, 1300, 22,
           "<b>第二步：推导完还有四条硬校验（前两条直接拒绝启动）</b>", "text_b")
    from dg import table
    rows = [
        ["batched < max_model_len 且关了分块", "ValueError", "不分块就必须一步装下整条序列"],
        ["batched < max_num_seqs", "ValueError", "一步连一个请求一个 token 都装不下"],
        ["batched > seqs × max_model_len", "只警告", "配置过松，不会炸"],
        ["long_prefill_token_threshold > max_model_len", "ValueError", "阈值比模型长度还大没意义"],
    ]
    table(p, X, y2, [("条件", 560), ("结果", 260), ("为什么", 600)], rows, row_h=36)

    # --- 它又决定了谁 -----------------------------------------------------
    y3 = y2 + 36 * 5 + 36
    p.text(X, y3 - 32, 1300, 22,
           "<b>第三步：这个参数反过来决定四样东西（联动闭环）</b>", "text_b")
    panel(p, X, y3, 700, 168, "① 图桶的上限", [
        "capture size 的候选先生成、再被 max_num_batched_tokens 截断；",
        "你把这个数调小 → 桶的上限跟着变小 → 大 batch 就落回 eager。",
        "本仓实例：一个包把 batched 设成 4096（TP=8 的 DeepSeek），桶上限自然更小。",
    ], "cloudd")
    panel(p, X + 760, y3, 664, 168, "② 每步能装多少 token", [
        "这就是「一步的 token 预算」——chunked prefill 切块、投机验证占 1+K，都从这里扣。",
        "调小 → 延迟更平滑但吞吐降；调大 → 吞吐高但长 prefill 会让 decode 卡顿。",
    ], "cloudp")

    y4 = y3 + 208
    panel(p, X, y4, 700, 148, "③ max_num_seqs（当它没被显式设时）", [
        "取「默认值」与 max_num_batched_tokens 的较小者；",
        "所以你把 batched 压小，seqs 会跟着被压小——两个一起变，容易误判是谁的原因。",
    ], "neutral")
    panel(p, X + 760, y4, 664, 148, "④ 投机解码的预算", [
        "开了 MTP 后每个请求要占 1+K 个位置，还要扣 draft 的额外槽，",
        "所以「同一份 batched 预算能装几个请求」会明显变少。",
    ], "fail")

    y5 = y4 + 188
    bridge(p, y5, "本页交接：调度参数是一条完整的推导链。下一步（⑥）走到边云 PD 那边——"
                  "那里有一批更硬的约束（两侧块大小必须相等、P 侧请求被强制改写）。")

    y6 = y5 + 76
    anchors(p, y6, [
        "七步推导：vllm/engine/arg_utils.py:2948-2957（入口）、:2730-2788（按显存/用途）、"
        ":2966-2970（throughput ×2）、:2976-2981（关分块抬高）、:2983-2998（多模态）、"
        ":3000-3006（截断）、:3014-3016（反推 seqs）",
        "四条硬校验：vllm/config/scheduler.py:296-307、:309-314、:316-322、:324-329",
        "桶被 batched 截断：vllm/config/vllm.py:2348-2349、:2364-2366",
        "每步 token 预算：vllm/v1/core/sched/scheduler.py:588-591、:670-672",
        "本仓实例：assets/c_home/bash/pds/fix_1120/prefill.sh:108（batched 4096）、"
        "其余包 8192",
    ], h=120)

    y7 = y6 + 154
    concl(p, y7,
          "<b>一句话记住这一页：</b>max_num_batched_tokens 是「一步的 token 预算」，"
          "默认值由引擎按显存/用途/模式/分块依次推导，推导完还会反推 max_num_seqs、"
          "截断图桶、扣掉投机占位——<b>看清它，才能解释「为什么我没改却变了」。</b>")

    legend(p, y7 + 92)
