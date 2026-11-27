"""④ 共享专家：一条「所有 token 都走」的旁路。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc

TITLE = "④ 共享专家：让 MoE 里有一条「谁都要走」的 dense 旁路"
SUB = ("一句话：DeepSeek 风格 MoE 除了 N 个路由专家，还有一个<b>所有 token 都必走</b>的共享专家"
       "（权重固定 1）。它承担「通用能力」，让路由专家可以真正特化——"
       "<b>代价是它不受稀疏保护，每步都要读</b>。")


def build(doc: Doc) -> None:
    p = doc.page("④ 共享专家", pw=1560, ph=1480)
    head(p, TITLE, SUB)
    X = 36

    # --- 结构 -------------------------------------------------------------
    y = 148
    p.text(X, y - 30, 1300, 22,
           "<b>第一步：一层 MoE 的完整结构（路由专家 + 共享专家）</b>", "text_b")
    p.box(X, y, 1424, 140,
          "<font style=\"font-size:10px\">"
          "　　　　　　　　　　x（token）<br>"
          "　　　　　　┌──────────┴──────────┐<br>"
          "　　　　<b>门卫</b>（打分 + top-k）　　　<b>共享专家</b>（必走，权重 = 1.0）<br>"
          "　　　　　　↓　　　　　　　　　　　　↓<br>"
          "　　　k 个路由专家的加权结果　　＋　　固定一路结果<br>"
          "　　　　　　└──────────┬──────────┘<br>"
          "　　　　　　　　　　　 ↓　输出 y</font>",
          "box", "cloudd")

    # --- 为什么需要 -------------------------------------------------------
    y2 = y + 180
    p.text(X, y2 - 30, 1300, 22,
           "<b>第二步：它解决什么问题（三个理由，按重要性排）</b>", "text_b")
    panel(p, X, y2, 460, 190, "① 给通用能力一条「固定容量」的路", [
        "路由专家每 token 只碰 k 个，容量有限且被热门专家挤占；",
        "语法、常见搭配这类「所有 token 都需要」的知识，",
        "如果也靠路由专家学，会被反复争抢、学不稳。",
        "共享专家给它们一条永远可用的通道。",
    ], "cloudp")
    panel(p, X + 482, y2, 460, 190, "② 让路由专家可以真正特化", [
        "有了兜底的通用通路，N 个路由专家就能分工更细、各自更窄。",
        "这也解释了为什么 MoE 模型的「每专家中间维」通常比 dense 的 FFN 宽度小很多。",
    ], "cloudd")
    panel(p, X + 964, y2, 460, 190, "③ 训练更稳", [
        "路由是离散选择（top-k），梯度稀疏、容易震荡；",
        "共享专家的梯度对每个 token 都存在，相当于给优化过程一个稳定的锚。",
    ], "neutral")

    # --- 实现上的三个细节 -------------------------------------------------
    y3 = y2 + 230
    p.text(X, y3 - 30, 1300, 22,
           "<b>第三步：实现上的三个细节（都能在仓里查到）</b>", "text_b")
    p.box(X, y3, 1424, 132,
          "<font style=\"font-size:10px\">"
          "① <b>不是一个「独立的专家」，而是一条更宽的 dense MLP</b>："
          "多个共享专家被合并成 <b>一个</b> 宽度 = 每专家宽 × 共享数 的 FFN。<br>"
          "② <b>计算量按「token 数」算，不按「token × k」</b>："
          "本仓性能模型里，路由专家的 FLOPs 用 token×k，共享专家用 token —— "
          "这就是「它不受稀疏保护」的定量表达。<br>"
          "③ <b>它不参与负载均衡</b>：EPLB 计算专家放置时会先把共享专家从总数里减掉，"
          "最后单独追加——因为它天然在所有卡上都存在一份（或按 dense 切分）。</font>",
          "box", "neutral")

    # --- 代价 -------------------------------------------------------------
    y4 = y3 + 172
    p.box(X, y4, 1424, 116,
          "<b>第四步：代价——它让 MoE「更像 dense」</b><br>"
          "<font style=\"font-size:10px\">"
          "共享专家是 dense 的：<b>每个 token 都走、每一步都要读它的权重</b>。"
          "所以共享专家越宽，MoE 的 decode 就越接近 dense（稀疏性带来的节省被吃回去）。<br>"
          "<font color=\"#c0392b\">推论：<b>共享专家宽度是一个「用 decode 速度换模型质量」的旋钮</b>——"
          "本仓配置里它通常是 1 个共享专家，而路由专家有上百个。</font></font>",
          "box", "fail")

    # --- 数字 -------------------------------------------------------------
    y5 = y4 + 156
    p.text(X, y5 - 30, 1300, 22,
           "<b>第五步：本仓两类 MoE 模型的默认结构（来自仓内自带的 config 定义）</b>",
           "text_b")
    cards = [
        ("DeepSeek-V3 风格（仓内默认）",
         "专家 <b>192</b> 个　·　top-k <b>8</b>　·　共享专家 <b>1</b> 个<br>"
         "每专家宽 <b>1536</b>　·　dense FFN 宽 <b>13312</b>", "cloudp"),
        ("GLM-5-Next 风格",
         "专家 <b>288</b> 个　·　top-k <b>7</b>　·　共享专家 <b>1</b> 个<br>"
         "每专家宽 <b>2048</b>", "cloudd"),
    ]
    cw = 700
    for i, (name, body, role) in enumerate(cards):
        cx = X + i * (cw + 24)
        p.box(cx, y5, cw, 96, "", "box", role)
        p.text(cx + 12, y5 + 8, cw - 24, 24, f"<b>{name}</b>", "text_b")
        p.text(cx + 12, y5 + 34, cw - 24, 58,
               f"<font style=\"font-size:10px\">{body}</font>", "text")
    p.text(X, y5 + 104, 1424, 40,
           "<font style=\"font-size:9px;color:#666666\">"
           "读法：专家数很多但每专家很窄（1536 vs dense 的 13312）；"
           "top-k 只有 8 → 每 token 激活约 8/192 ≈ 4.2% 的专家参数。</font>", "subtitle")

    y6 = y5 + 150
    bridge(p, y6, "本页交接：结构齐了。下一步（⑤）讲「专家怎么切到多张卡」——这就是专家并行（EP）。")

    y7 = y6 + 76
    anchors(p, y7, [
        "共享专家的构造（宽度 = 每专家宽 × 共享数）：vllm/model_executor/models/hy_v3.py:164-175",
        "「所有 token 都走」的定量证据（FLOPs 用 token 而非 token×k）："
        "vllm/v1/metrics/perf.py:1253-1255（注释原文 all shared experts run for every token）",
        "融合放置时把共享专家补进 top-k、权重恒 1："
        "vllm-ascend/vllm_ascend/ops/fused_moe/routed_experts.py:665-680",
        "EPLB 计算时先减掉共享专家：vllm-ascend/vllm_ascend/eplb/core/eplb_utils.py:51-55、:72-74",
        "默认配置字段：vllm/transformers_utils/configs/hy_v3.py:62-71、:124-135"
        "（num_experts=192 / per_tok=8 / shared=1 / expert_hidden_dim=1536 / dense 13312 / "
        "first_k_dense_replace=1）",
        "GLM-5-Next：vllm-ascend/vllm_ascend/models/glm5next/config.py:30-35"
        "（288 专家 / top-7 / 1 共享 / 2048）",
    ], h=136)

    y8 = y7 + 170
    concl(p, y8,
          "<b>一句话记住这一页：</b>共享专家 = 一条所有 token 必走的 dense 旁路，"
          "作用是「兜通用能力 + 让路由专家能特化 + 训练更稳」；"
          "<b>代价是它不受稀疏保护，越宽 MoE 越像 dense。</b>")

    legend(p, y8 + 92, extra_fill=("共享专家（必走）", "cloudd"))
