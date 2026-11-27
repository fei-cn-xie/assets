"""④ 一次 draft step 的十步（按代码顺序）。"""
from common import head, concl, anchors, legend, bridge
from dg import Doc

TITLE = "④ 一次带 draft 的 step：十步走完（按代码顺序）"
SUB = ("一句话：<b>调度器一次排 1+N 个位置 → 一次前向算出这 N 行验证 logits + 1 行 bonus → "
       "拒绝采样逐位判 → 回退 KV → 用本步隐状态起草下一步</b>。")


def row(p, y, no, title, body, role, h=62):
    p.box(36, y, 1424, h,
          f"<b>{no}　{title}</b><br>"
          f"<font style=\"font-size:9px\" color=\"#333333\">{body}</font>",
          "box", role)
    return y + h + 8


def build(doc: Doc) -> None:
    p = doc.page("④ draft step 十步", pw=1560, ph=1640)
    head(p, TITLE, SUB)
    X = 36

    y = 150
    p.text(X, y - 32, 1300, 22,
           "<b>上：调度器与 worker（1-5）　　下：验证与回写（6-10）</b>", "text_b")

    y = row(p, y, "①", "调度器一次排 1+N 个位置",
            "num_new_tokens 一般就是 1+N（N = num_speculative_tokens），"
            "再被「剩余上下文长度」收紧。", "cloudp")
    y = row(p, y, "②", "按 1+N 分配 KV，并给 drafter 多留 N 个槽",
            "allocate_slots(..., num_lookahead_tokens=N)：因为 drafter 会往 target 之后写 KV，"
            "必须多预留（否则越界）。", "fail")
    y = row(p, y, "③", "把草案写进本步输出，并清空请求侧缓存",
            "scheduled_spec_decode_tokens[req] = 草案；request.spec_token_ids 清空；"
            "padding 的位置填 -1（表示「这些位置没有草案」）。", "neutral")
    y = row(p, y, "④", "worker 把草案灌进输入批（含 positions / slot_mapping）",
            "positions = 已算 token 数 + 位置偏移；一次前向就要覆盖这 1+N 行。", "cloudd")
    y = row(p, y, "⑤", "算 logits 行索引：每请求 N 行验证 + 1 行 bonus",
            "logits_indices / target_logits_indices / bonus_logits_indices 三组索引，"
            "把「哪一行是哪个请求的哪个位置」映射清楚。", "cloudd", h=68)
    y = row(p, y, "⑥", "一次前向：target 算出 1+N 行 logits",
            "这一步就是「一次前向多干活」的实体：权重读一遍，算出所有候选位置的分布。", "cloudd")
    y = row(p, y, "⑦", "先采 bonus（最后一行）",
            "bonus 只用 target 分布采（可带 temperature/top-p），是「全中时白送」的那个 token。",
            "cloudp")
    y = row(p, y, "⑧", "拒绝采样逐位判：贪心 or 随机",
            "随机：u < p_target/p_draft 才接受；贪心：与 target 的 argmax 相等才接受。"
            "一旦某位被拒，后面全部作废。", "cloudp", h=68)
    y = row(p, y, "⑨", "被拒位置补一个 token（recovered），输出定长 [B, N+1]",
            "recovered 从残差分布 max(p_target − p_draft, 0) 归一化后采样；"
            "输出数组空位填 -1，同步路径再把 -1 截掉。", "cloudp", h=68)
    y = row(p, y, "⑩", "回退 KV，并用本步隐状态起草下一步",
            "num_computed_tokens -= 被拒个数；草案由 drafter.propose() 用本步 hidden states 生成，"
            "异步拷回后交给 scheduler 缓存成 spec_token_ids。", "data", h=68)

    y2 = y + 4
    bridge(p, y2, "本页交接：第 ⑧ 步「拒绝采样」是整套机制的数学核心——"
                  "下一步（⑤）把贪心与随机两条分支讲透，包括为什么它不改变输出分布。")

    y3 = y2 + 76
    anchors(p, y3, [
        "①②③ 调度侧：vllm/v1/core/sched/scheduler.py:663-681（1+N 宽度）、:741-745"
        "（lookahead 槽）、:823-839（写 draft 并清缓存）、:1330-1334（-1 填充）",
        "④⑤ 灌输入与索引：vllm/v1/worker/gpu_model_runner.py:1521-1533、:2245-2280、"
        ":2833-2906",
        "⑥ 一次前向：同文件 :4188-4271",
        "⑦⑧⑨ 验证：vllm/v1/sample/rejection_sampler.py:140-162（bonus）、:188-200（入口）、"
        ":719-773（贪心）、:822-893（随机）、:1043-1077（recovered）",
        "⑩ 回退与起草：scheduler.py:2041-2059、gpu_model_runner.py:4625-4639、:5020-5297",
    ], h=112)

    y4 = y3 + 146
    concl(p, y4,
          "<b>一句话记住这一页：</b>一次带 draft 的 step = 「排 1+N 个位置 → 一次前向 → "
          "逐位判 + 补位 → 回退被拒的 KV → 起草下一步」；"
          "<b>N 个草案里能被接受的个数，决定了这一步实际白赚几个 token。</b>")

    legend(p, y4 + 92, extra_fill=("起草（draft）相关", "cloudp"))
