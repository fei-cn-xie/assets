"""③ 草案从哪来：四种来源与 vLLM 里的实现类。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc, table

TITLE = "③ 草案从哪来：四种来源，各有各的代价"
SUB = ("一句话：draft 可以来自<b>同模型的额外头（MTP/EAGLE）、独立小模型、n-gram 查表、"
       "或并行多头（Medusa）</b>；选哪种决定了「省不省显存」与「准不准」。")


def build(doc: Doc) -> None:
    p = doc.page("③ 草案的来源", pw=1560, ph=1480)
    head(p, TITLE, SUB)
    X = 36

    y = 152
    p.text(X, y - 34, 1300, 22,
           "<b>第一步：四种来源（按「要不要额外权重」排序）</b>", "text_b")
    rows = [
        ["① 同模型自带 MTP 头", "target 里已经有 nextn 层（权重和模型一起下载）",
         "不需要额外权重；复用 target 的 embedding/lm_head/KV",
         "method=mtp"],
        ["② EAGLE 类小头", "吃 target 的隐状态，自回归猜 N 步",
         "准；但仍需一份小头权重", "method=eagle / eagle3"],
        ["③ 独立小模型", "另起一个小模型自回归起草",
         "可插拔、泛化好；但要另占显存与一份 KV", "method=draft_model"],
        ["④ n-gram / suffix 查表", "在同段文本里找重复片段，直接把后面抄过来",
         "零模型零显存；文本有重复时极准，否则几乎无效", "method=ngram / suffix"],
    ]
    table(p, X, y, [("来源", 260), ("怎么出草案", 420), ("优缺点", 560),
                    ("配置里的 method", 180)], rows, row_h=46)

    # --- 代码里的选择点 ---------------------------------------------------
    y2 = y + 46 * 5 + 40
    p.text(X, y2 - 34, 1300, 22,
           "<b>第二步：vLLM 里「选谁当起草者」就是一个 if/elif（没有 ProposerManager）</b>",
           "text_b")
    p.box(X, y2, 1424, 104,
          "<font style=\"font-size:10px\">"
          "在 model runner 初始化时按 method 直接构造："
          "<font style=\"font-size:9px\">EagleProposer / MedusaProposer / "
          "DraftModelProposer / NgramProposer / SuffixDecodingProposer …</font><br>"
          "同一个地方还构造 <font style=\"font-size:9px\">rejection_sampler</font>"
          "（验证器）。所以「起草」与「验证」是<b>成对</b>装配的。<br>"
          "<font color=\"#c0392b\">注意：<font style=\"font-size:9px\">method='mtp'</font> "
          "与他人共用 EAGLE 那条实现路径（代码里 mtp 被归到 eagle 一类）。</font></font>",
          "box", "neutral")

    # 四象限
    y3 = y2 + 144
    p.text(X, y3 - 34, 1300, 22,
           "<b>第三步：怎么选（两把尺子：显存 vs 接受率）</b>", "text_b")
    panel(p, X, y3, 700, 168, "省显存优先 → MTP / EAGLE", [
        "复用 target 的 embedding、lm_head，甚至共享同一份 KV cache；",
        "所以额外显存开销很小，而且接受率通常最高（因为它的隐藏状态与 target 同源）。",
        "代价：只有「模型自带这个头」或「能找到配套小头」时才能用。",
    ], "cloudd")
    panel(p, X + 760, y3, 664, 168, "零成本优先 → n-gram / suffix", [
        "完全不用模型，纯字符串匹配；对「代码补全 / 大段引用 / 格式化输出」特别有效。",
        "但遇到开放式生成几乎全被拒——被拒也要付验证成本，所以可能是负收益。",
        "适合做「能省则省」的兜底策略，而不是主策略。",
    ], "neutral")

    y4 = y3 + 208
    panel(p, X, y4, 1424, 108, "独立小模型（draft_model）：为什么它反而少见", [
        "要另带一份权重 + 一份 KV（显存直接变贵），而且小模型的分布与大模型差得远、接受率偏低；",
        "但它有两个不可替代的场景：<b>① 大模型族没有自带 MTP 头</b>；"
        "<b>② 需要「跨模型复用」同一个 drafter</b>（一个 drafter 服务多种 target）。",
        "本仓部署里用的是 MTP（见 ⑨），不是这个。",
    ], "cloudp")

    y5 = y4 + 148
    bridge(p, y5, "本页交接：知道草案从哪来之后，就该看「它怎么被验证」——"
                  "这是整套机制的数学核心。下一步（④）走一遍完整的 10 步流程。")

    y6 = y5 + 76
    anchors(p, y6, [
        "proposer 的构造（if/elif 装配 + rejection_sampler）："
        "vllm/v1/worker/gpu_model_runner.py:629-704",
        "method 判定谓词（use_eagle / uses_draft_model 等）：vllm/config/speculative.py:1909-1942",
        "各实现类：vllm/v1/spec_decode/eagle.py:10、medusa.py:18、draft_model.py:19、"
        "ngram_proposer.py:12、suffix_decoding.py:9",
        "MTP 被归入 eagle 一类：vllm/config/speculative.py:1909-1913",
        "MTP 层复用 target 权重（共享 embedding/lm_head）："
        "vllm/v1/spec_decode/llm_base_proposer.py:1406、:1517",
    ], h=112)

    y7 = y6 + 146
    concl(p, y7,
          "<b>一句话记住这一页：</b>draft 的四种来源本质是「用多少额外资源换多高的接受率」："
          "MTP/EAGLE 最省也最准（复用 target），独立小模型最通用但最贵，"
          "n-gram 零成本但只在有重复时有效。")

    legend(p, y7 + 92)
