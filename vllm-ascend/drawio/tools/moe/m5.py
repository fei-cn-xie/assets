"""⑤ 专家并行（EP）：把专家切到多张卡上。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc

TITLE = "⑤ 专家并行（EP）：把「很多专家」切到多张卡，但总量一点没少"
SUB = ("一句话：EP 就是把 N 个专家<b>按卡切分</b>，每卡只放 N/ep 个；"
       "省的是「单卡必须放下全部专家」，<b>不省总显存</b>——"
       "代价是 token 得跨卡去找自己的专家。")


def build(doc: Doc) -> None:
    p = doc.page("⑤ 专家并行", pw=1560, ph=1520)
    head(p, TITLE, SUB)
    X = 36

    # --- 切分示意 ---------------------------------------------------------
    y = 148
    p.text(X, y - 30, 1300, 22,
           "<b>第一步：8 个专家切到 2 张卡（每卡 4 个）</b>", "text_b")
    p.box(X, y, 690, 172,
          "<b>不切（每卡都放全部专家）</b><br><br>"
          "<font style=\"font-size:10px\">"
          "卡 0：专家 0 1 2 3 4 5 6 7<br>"
          "卡 1：专家 0 1 2 3 4 5 6 7<br><br>"
          "每张卡都要放下全部专家 → 单卡显存压力最大；<br>"
          "但每卡都能本地算，<b>不需要搬 token</b>。</font>",
          "box", "fail")
    p.box(X + 730, y, 694, 172,
          "<b>切分（每卡一半专家）</b><br><br>"
          "<font style=\"font-size:10px\">"
          "卡 0：专家 0 1 2 3<br>"
          "卡 1：专家 4 5 6 7<br><br>"
          "单卡显存压力减半；<br>"
          "但 token 若被分到「别人家的专家」，就必须<b>跨卡送过去</b>（见 ⑥）。</font>",
          "box", "cloudd")

    # --- 一个关键澄清 -----------------------------------------------------
    y2 = y + 212
    p.box(X, y2, 1424, 120,
          "<b>第二步：必须澄清的一点——EP 不减少总显存</b><br>"
          "<font style=\"font-size:10px\">"
          "切分只是把「一批专家」摊到多张卡上：<b>全部专家仍然要在整个集群里驻留一份</b>。"
          "所以 EP 解决的是「<b>单卡放不下</b>」，不是「<b>总共太多</b>」。<br>"
          "本仓代码里能直接看到这一点：每张卡维护一张「全局专家 → 本卡槽位」的映射，"
          "非本卡的专家映射为 <b>−1</b>（表示「这个专家不在我这」）。<br>"
          "<font color=\"#c0392b\">推论：EP 之后「可用 KV 显存」并不会因为切分而变多——"
          "它和「专家数 × 每专家大小」一起，仍然是显存账里的第一大户（占权重 85~90%）。</font></font>",
          "box", "neutral")

    # --- EP 与 TP/DP 的关系 ----------------------------------------------
    y3 = y2 + 160
    p.text(X, y3 - 30, 1300, 22,
           "<b>第三步：EP 和 TP / DP 的关系（三个容易混的概念）</b>", "text_b")
    from dg import table
    rows = [
        ["TP（张量并行）", "切「一层内的矩阵」", "每层都要通信（all_reduce）", "dense 与 MoE 都用"],
        ["EP（专家并行）", "切「专家」", "token 跨卡搬（前后各一次）", "<b>只有 MoE 才有</b>"],
        ["DP（数据并行）", "复制整模型，各跑各的请求", "推理时几乎不通信", "两者都能用"],
    ]
    table(p, X, y3, [("概念", 200), ("切什么", 340), ("通信代价", 460), ("谁能用", 424)],
          rows, row_h=38)
    p.box(X, y3 + 38 * 4 + 16, 1424, 96,
          "<font style=\"font-size:10px\">"
          "本仓文档给的换算是 <b>EP 规模 = TP 规模 × DP 规模</b>——"
          "也就是说：单机 8 卡 TP=8 时开 EP，等于把这 8 张卡都当成「专家池」；"
          "而 DP=2 时，两个副本的卡也会计入 EP 池。<br>"
          "<font color=\"#c0392b\">所以开关只有一句 <font style=\"font-size:9px\">--enable-expert-parallel</font>，"
          "但实际拓扑由 TP/DP 一起决定。</font></font>",
          "box", "cloudp")

    # --- 为什么切了还要「两张表」 -----------------------------------------
    y4 = y3 + 38 * 4 + 152
    p.text(X, y4 - 30, 1300, 22,
           "<b>第四步：切分后为什么需要「逻辑专家 → 物理专家」两层映射</b>",
           "text_b")
    p.panel if False else None
    panel(p, X, y4, 700, 168, "原因：专家的位置可能被「搬动」", [
        "模型里定义的专家编号（逻辑）与「它此刻在哪张卡、哪个槽位」（物理）是两回事。",
        "因为 ⑦ 页要讲的负载均衡会把热门专家复制几份、挪到别的卡上。",
        "所以路由算完之后，还要过一次「逻辑 → 物理」映射表。",
    ], "data")
    panel(p, X + 760, y4, 664, 168, "本仓的实现", [
        "有一张映射表把逻辑专家号映射到物理专家号；",
        "未启用负载均衡时，它基本是恒等映射（第 i 个专家就在第 i 个槽位）。",
        "启用之后，同一个逻辑专家可能对应多个物理副本。",
    ], "cloudd")

    y5 = y4 + 208
    bridge(p, y5, "本页交接：切了专家就要搬 token。下一步（⑥）看搬的三种实现——"
                  "注意本仓默认<b>不是</b> all-to-all。")

    y6 = y5 + 76
    anchors(p, y6, [
        "每卡只放 global_num_experts // ep_size 个、非本卡映射为 −1："
        "vllm/model_executor/layers/fused_moe/expert_map_manager.py:22-69",
        "EP 规模 = TP × DP（本仓文档原文）：vllm/docs/serving/expert_parallel_deployment.md:36-40",
        "逻辑→物理映射与负载记录：vllm-ascend/vllm_ascend/ops/fused_moe/eplb.py:10、:37、:87、:99",
        "专家占权重字节 85~90%：vllm/model_executor/model_loader/ep_weight_filter.py:1-9",
        "EP 开关在代码里改变什么：`--enable-expert-parallel` 传导到 MoE 通信方法选择，"
        "见 ⑥ 页的选择链（ascend_forward_context.py:373-425）",
    ], h=120)

    y7 = y6 + 154
    concl(p, y7,
          "<b>一句话记住这一页：</b>EP = 把专家切到多卡，省的是「单卡放得下」，"
          "不省总显存；代价是把「本地计算」换成「跨卡搬 token」，"
          "并且多出一层「逻辑专家 → 物理专家」的映射。<b>没有 MoE 就没有 EP。</b>")

    legend(p, y7 + 92, extra_fill=("映射表/数据面", "data"))
