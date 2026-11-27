"""③ MoE 的三本账：显存、通信、负载。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc

TITLE = "③ MoE 推理的三本账：显存、通信、负载——以及 decode 为什么更吃亏"
SUB = ("一句话：MoE 把 compute 省下来了，代价转移到了<b>三处</b>："
       "全部专家都要驻留显存、token 要跨卡搬（all-to-all）、热门专家会把某个卡拖成瓶颈。")


def build(doc: Doc) -> None:
    p = doc.page("③ MoE 三本账", pw=1560, ph=1560)
    head(p, TITLE, SUB)
    X = 36

    # --- 三本账 -----------------------------------------------------------
    y = 146
    p.text(X, y - 30, 1300, 22,
           "<b>第一步：三本账（每本都有本仓可查的数字或注释）</b>", "text_b")
    panel(p, X, y, 460, 210, "账一：显存", [
        "全部专家都要驻留——切到多卡只是「摊开」，总量不降。",
        "本仓有一个很硬的说法：<b>专家约占权重字节的 85~90%</b>。",
        "所以 MoE 模型的显存几乎全花在专家上；",
        "而 KV cache 只能吃剩下的零头（与入门套 ⑦ 的显存账一致）。",
    ], "fail")
    panel(p, X + 482, y, 460, 210, "账二：通信", [
        "专家切到多卡后，token 要去别的卡上的专家 → 专家计算<b>前后各一次 all-to-all</b>。",
        "搬的内容 = 「token 数 × k」行 hidden。",
        "关键：接收缓冲按<b>最坏情况</b>分配（每卡最大批 × EP 规模），",
        "所以<b>小 batch 也要付满额开销</b>。",
    ], "data")
    panel(p, X + 964, y, 460, 210, "账三：负载", [
        "热门专家忙死、冷门专家闲着——本仓文档原文说实际分布可能<b>高度偏斜</b>。",
        "后果：某个 EP rank 成了瓶颈，其他 rank 在等它。",
        "性能模型里甚至留了个注释：<font style=\"font-size:9px\">"
        "FIXME: Assume perfect load balancing for now</font>"
        "——即「目前只能假设负载均衡」。",
    ], "cloudp")

    # --- decode 为什么更吃亏 ----------------------------------------------
    y2 = y + 250
    p.text(X, y2 - 30, 1300, 22,
           "<b>第二步：decode 为什么比 prefill 更吃亏（四条，逐条给依据）</b>",
           "text_b")
    p.box(X, y2, 1424, 140,
          "<font style=\"font-size:10px\">"
          "① <b>要读的专家权重数量随批大小变化</b>：本仓性能模型里，dense 每步读「整份 FFN 权重」"
          "（与 token 数无关）；MoE 每步读的是 <b>min(token 数 × k, 专家总数)</b> 份。"
          "批大时摊薄，批小时反而读得零碎。<br>"
          "② <b>访存不连续</b>：专家权重是分散的多块，单次能读到的有效行少，带宽利用率低。<br>"
          "③ <b>两次 all-to-all 的固定开销</b>：小 batch 时通信延迟盖过计算时间。<br>"
          "④ <b>共享专家不受稀疏保护</b>：它是 dense 的，每个 token 必走、每步必读（见 ④ 页）。</font>",
          "box", "fail")

    # --- 一个「不能直接照搬」的推论 ---------------------------------------
    y3 = y2 + 180
    p.box(X, y3, 1424, 128,
          "<b>第三步：纠正一个常见推论（这条很重要）</b><br>"
          "<font style=\"font-size:10px\">"
          "常见说法：「MoE 每 token 只走 k 个专家，所以 decode 一定比 dense 省」。<br>"
          "<font color=\"#c0392b\">这个推论不成立。</font>"
          "因为 <b>专家总参数量远大于 dense FFN</b>（本仓默认配置里专家中间维与 dense 宽度差近 9 倍，"
          "总专家数上百），而 decode 的批很小 —— "
          "<b>min(批 × k, 专家数) 往往仍远大于 1</b>，加上 attention 与共享专家照样要读，"
          "实际读权重总量通常<b>仍显著高于</b>同 hidden 的 dense 层。<br>"
          "<font color=\"#666666\">（前半句的依据是本仓性能模型公式；"
          "「通常仍高于 dense」是通用结论，图上标为推导。）</font></font>",
          "box", "neutral")

    # --- 所以怎么办 -------------------------------------------------------
    y4 = y3 + 168
    p.text(X, y4 - 30, 1300, 22,
           "<b>第四步：三本账各自对应的解法（后面几页逐个展开）</b>", "text_b")
    from dg import table
    rows = [
        ["显存紧", "把专家切到多卡（EP）；量化压专家权重", "⑤ 页、⑥ 页"],
        ["通信贵", "按阶段选通信实现（prefill 走高吞吐、decode 走低延迟）", "⑥ 页"],
        ["负载偏斜", "训练期加辅助损失 / 路由偏置；推理期换位 + 冗余专家（EPLB）", "⑦ 页"],
        ["共享专家太贵", "减小共享专家宽度", "④ 页"],
    ]
    table(p, X, y4, [("问题", 220), ("手段", 820), ("在哪页", 384)], rows,
          row_h=36)

    y5 = y4 + 36 * 5 + 36
    bridge(p, y5, "本页交接：三本账讲完。下一步（④）先把「共享专家」这个反直觉的东西讲清——"
                  "它让 MoE 更像 dense，而不是更像 MoE。")

    y6 = y5 + 76
    anchors(p, y6, [
        "专家占权重 85~90%：vllm/model_executor/model_loader/ep_weight_filter.py:1-9（原文）",
        "dense vs MoE 每步读权重的公式：vllm/v1/metrics/perf.py:1289-1347"
        "（MoE 用 num_activated_experts = min(T*E, num_experts) :1311；"
        "假设均衡的 FIXME 注释 :1310）",
        "专家计算前后两次 all-to-all：vllm-ascend/.../fused_moe/moe_comm_method.py:263-271（原文）",
        "接收缓冲按最坏情况分配：同文件 :416-422（num_max_tokens_per_rank × ep_world_size × min(topk, …)）",
        "「小 batch 也付满额开销」：token_dispatcher.py:126-133（global_bs = max_bs_per_rank × ep_world_size）",
        "实际分布高度偏斜（原文）：vllm/docs/serving/expert_parallel_deployment.md:138",
    ], h=136)

    y7 = y6 + 170
    concl(p, y7,
          "<b>一句话记住这一页：</b>MoE 省的是「算」，付出的是「显存 + 通信 + 负载不均」；"
          "而这笔交易在 <b>prefill（大批）时划算、在 decode（小批）时要还回去</b>——"
          "所以 MoE 的调优几乎都围绕这三本账。")

    legend(p, y7 + 92, extra_fill=("通信/搬运这条账", "data"))
