"""⑧ 一张卡放不下：TP / PP / DP / EP 四种切法的动机与代价。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc, table

TITLE = "⑧ 一张卡放不下：四种并行，各切什么、各通信什么"
SUB = ("一句话：<b>TP 切一层内部的矩阵</b>、<b>PP 切层</b>、<b>DP 复制整份跑不同请求</b>、"
       "<b>EP 切专家</b>；每切一刀都要付通信代价，所以「切哪种」由瓶颈决定。")

CFG = 10


def build(doc: Doc) -> None:
    p = doc.page("⑧ 四种并行", pw=1560, ph=1500)
    head(p, TITLE, SUB)
    X = 36

    # --- 四种切法 ---------------------------------------------------------
    y = 152
    p.text(X, y - 34, 1300, 22,
           "<b>第一步：四种切法（每个方框是一张卡）</b>", "text_b")

    # TP
    p.text(X, y, 340, 20, "<b>TP（张量并行）：切一层的矩阵</b>", "text_b")
    for i in range(4):
        p.box(X + i * (76 + CFG), y + 24, 76, 62,
              f"<b>卡{i}</b><br><font style=\"font-size:9px\">1/4 个头</font>",
              "box_center", "cloudp")
    p.text(X, y + 92, 400, 40,
           "<font style=\"font-size:9px;color:#666666\">"
           "每层一次 all_reduce（行并行的输出要相加）</font>", "subtitle")

    # PP
    p.text(X, y + 140, 340, 20, "<b>PP（流水并行）：切层</b>", "text_b")
    for i, lab in enumerate(["层 0-7", "层 8-15", "层 16-23", "层 24-31"]):
        p.box(X + i * (76 + CFG), y + 164, 76, 62,
              f"<b>卡{i}</b><br><font style=\"font-size:9px\">{lab}</font>",
              "box_center", "cloudd")
    p.text(X, y + 232, 400, 40,
           "<font style=\"font-size:9px;color:#666666\">"
           "卡 0 算完把激活传给卡 1（点对点）；要「填满流水」需多批</font>",
           "subtitle")

    # DP
    p.text(X + 520, y, 400, 20, "<b>DP（数据并行）：复制整份，各跑各的请求</b>",
           "text_b")
    for i in range(2):
        p.box(X + 520 + i * 200, y + 24, 180, 62,
              f"<b>卡组{i}（整模型）</b><br>"
              f"<font style=\"font-size:9px\">独立调度、独立 KV</font>",
              "box_center", "data")
    p.text(X + 520, y + 92, 420, 40,
           "<font style=\"font-size:9px;color:#666666\">"
           "推理时互不通信（各服务各的请求），只需前端分发</font>", "subtitle")

    # EP
    p.text(X + 520, y + 140, 400, 20, "<b>EP（专家并行）：切 MoE 的专家</b>",
           "text_b")
    for i in range(4):
        p.box(X + 520 + i * (76 + CFG), y + 164, 76, 62,
              f"<b>卡{i}</b><br><font style=\"font-size:9px\">专家子集</font>",
              "box_center", "fail")
    p.text(X + 520, y + 232, 460, 40,
           "<font style=\"font-size:9px;color:#666666\">"
           "每个 token 只走少数专家 → 需要 all-to-all 把 token 送到对应卡</font>",
           "subtitle")

    # --- 对照表 -----------------------------------------------------------
    y2 = y + 292
    p.text(X, y2 - 34, 1300, 22,
           "<b>第二步：四种切法的对照表（切什么、通信什么、什么时候用）</b>",
           "text_b")
    rows = [
        ["TP", "一层内部的权重矩阵（按输出维/输入维）", "每层 all_reduce / all_gather",
         "单层都放不下、要低延迟", "并行度 ≤ 单机卡数（通信太贵）"],
        ["PP", "按层切段", "段间点对点传激活", "层数很多、跨机也行", "有流水气泡，需要多批填充"],
        ["DP", "不切，整份复制", "推理时几乎不用通信", "要提吞吐/多机扩容",
         "每份都占一份权重显存"],
        ["EP", "MoE 的专家", "all-to-all（token 路由）", "MoE 模型、专家太多",
         "负载不均会拖慢整批"],
    ]
    table(p, X, y2, [("", 90), ("切什么", 380), ("通信", 320), ("什么时候用", 340),
                     ("代价", 300)], rows, row_h=40)

    # --- 怎么选 -----------------------------------------------------------
    y3 = y2 + 40 * 5 + 36
    panel(p, X, y3, 700, 152, "选择的直觉（三个问题）", [
        "① <b>一层都放不下？</b> → 只能 TP（切层内的矩阵）。",
        "② <b>放得下但想要更高吞吐？</b> → DP（复制几份，各跑各的请求）。",
        "③ <b>层太多、单机放不下整模型？</b> → PP（或者 TP+PP 组合）。",
        "MoE 模型额外问一句：专家总数是否超过单卡显存 → EP。",
    ], "neutral")
    panel(p, X + 760, y3, 664, 152, "为什么本仓 PD 场景是「TP 或 DP」而不是 PP", [
        "PD 分离把 prefill 与 decode 拆到不同实例，"
        "每个实例只需要「跑得动一层 + 有足够 KV」；",
        "而 decode 阶段最怕的是<b>每层都要通信</b>（TP 的 all_reduce 在带宽瓶颈下更贵），",
        "所以本仓 decode 侧更偏好 DP（多份独立引擎、互不通信）。",
    ], "cloudp")

    y4 = y3 + 192
    bridge(p, y4, "本页交接：到这里，推理系统的四大需求（算力、并发、显存、多卡）都讲齐了。"
                  "下一步（⑨）把它们一次性映射到 vLLM 的组件上。")

    y5 = y4 + 76
    anchors(p, y5, [
        "并行字段定义：vllm/config/parallel.py:122（pipeline）、:124（tensor）、"
        ":129（data）、:167（enable_expert_parallel）",
        "world_size 与后端选择：vllm/config/parallel.py:887-895、:982-1027",
        "TP 的通信位置（行并行末尾 all_reduce）：vllm/model_executor/layers/linear.py:1775-1776",
        "进程数与并行的关系：见「vLLM进程与线程」套页 ②③",
    ], h=96)

    y6 = y5 + 128
    concl(p, y6,
          "<b>一句话记住这一页：</b>TP 切矩阵（每层通信）、PP 切层（段间通信、有气泡）、"
          "DP 复制（几乎不通信但吃显存）、EP 切专家（all-to-all）；"
          "<b>切哪种取决于「你缺的是显存、算力还是通信预算」。</b>")

    legend(p, y6 + 92)
