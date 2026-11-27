"""⑦ 回退：为什么「同一份代码在不同机器上性能差很多」。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc

TITLE = "⑦ 六层闸门：融合算子是怎么被一层层关掉的"
SUB = ("一句话：本仓的算子路径不是「写成什么样就跑成什么样」，"
       "而是<b>六道闸门逐级判断</b>——每一道不过，就降一档实现。"
       "所以「同一份代码在不同机器上性能差很多」通常是<b>闸门关的位置不同</b>。")


def build(doc: Doc) -> None:
    p = doc.page("⑦ 六层闸门", pw=1560, ph=1560)
    head(p, TITLE, SUB)
    X = 36

    # --- 六道闸门 ---------------------------------------------------------
    y = 146
    p.text(X, y - 30, 1300, 22,
           "<b>第一步：六道闸门（从上到下判，任一不过就降档）</b>", "text_b")
    gates = [
        ("① 自定义算子库", "能不能加载（版本/编译不匹配就整体禁用）", "fail"),
        ("② 硬件能力", "这台机器的芯片代次支持这个能力吗", "fail"),
        ("③ 运行时开关", "用户/环境变量允许吗（有些默认关）", "cloudp"),
        ("④ 形状与 dtype", "头数、维度、dtype 组合符合吗", "cloudp"),
        ("⑤ 布局与并行", "NZ 布局、TP 规模、CP 开关符合吗", "cloudd"),
        ("⑥ 走融合路径", "全部通过 → 用融合算子", "cloudd"),
    ]
    cw = 224
    for i, (name, note, role) in enumerate(gates):
        cx = X + i * (cw + 12)
        p.box(cx, y, cw, 96,
              f"<b>{name}</b><br>"
              f"<font style=\"font-size:9px;color:#333333\">{note}</font>",
              "box_center", role)
        if i:
            p.edge(src_pt=(cx - 10, y + 48), dst_pt=(cx - 2, y + 48), st="arrow",
                   role="http")
    p.text(X, y + 104, 1424, 24,
           "<font style=\"font-size:9px;color:#666666\">"
           "任一道不过 → 降一档：融合版 → 厂商版 → 基础版；功能不变，只是慢。</font>",
           "subtitle")

    # --- 三档实现 ---------------------------------------------------------
    y2 = y + 146
    p.text(X, y2 - 30, 1300, 22,
           "<b>第二步：同一件事的三档实现（以「相加 + 归一化」为例）</b>", "text_b")
    tiers = [
        ("A 档：图级再融合", "编译期把「相加 + 归一化 + 量化」合成一个算子",
         "需要：特定 dtype + 特定形状 + 融合 pass 开启", "cloudd"),
        ("B 档：厂商融合算子", "「相加 + 归一化 + bias」一次完成",
         "需要：芯片支持 + 自定义算子库加载成功", "cloudp"),
        ("C 档：分步基础算子", "相加一个算子、归一化一个算子",
         "总能跑（这是保底档）", "fail"),
    ]
    for i, (name, body, cond, role) in enumerate(tiers):
        cy = y2 + i * 82
        p.box(X, cy, 1424, 66, "", "box", role)
        p.text(X + 12, cy + 6, 220, 24, f"<b>{name}</b>", "text_b")
        p.text(X + 240, cy + 6, 700, 52,
               f"<font style=\"font-size:10px\">{body}</font>", "text")
        p.text(X + 960, cy + 6, 452, 52,
               f"<font style=\"font-size:9px;color:#333333\">{cond}</font>", "text")

    # --- 怎么发现自己在哪一档 ---------------------------------------------
    y3 = y2 + 3 * 82 + 14
    p.text(X, y3 - 30, 1300, 22,
           "<b>第三步：怎么发现「我到底在哪一档」（排障关键）</b>", "text_b")
    p.box(X, y3, 1424, 132,
          "<font style=\"font-size:10px\">"
          "① <b>看启动日志</b>：自定义算子库加载失败会有一条明确的 warning"
          "（大意是「自定义算子注册失败，全部禁用」）——这是最硬的一条。<br>"
          "② <b>看融合 pass 的日志</b>：图级融合被关掉时通常有 INFO/debug 说明原因"
          "（例如「head_dim 不是 128，QKNorm+RoPE 融合不启用」）。<br>"
          "③ <b>看代码里的前置条件函数</b>：融合路径前的检查函数会把「为什么不行」逐条列出来"
          "（TP 规模、dtype、bias、K 范围等）。<br>"
          "④ <b>对比相同配置的两台机器</b>：如果一台快一台慢，先比「芯片代次 + 算子库版本」，"
          "再比形状（seq len、头数、batch）。</font>",
          "box", "neutral")

    # --- 一个真实注意点 ---------------------------------------------------
    y4 = y3 + 172
    p.box(X, y4, 1424, 116,
          "<b>第四步：一个容易被忽略的「为兼容而故意不融合」</b><br>"
          "<font style=\"font-size:10px\">"
          "某些平台上有一种反向保护：当矩阵退化成「1×1」这种极小形状时，"
          "代码会<b>故意保持普通布局</b>，因为分形布局的矩阵算子在这种形状上会被直接拒绝。<br>"
          "<font color=\"#c0392b\">所以「为什么这里没转成最优布局」的答案可能是："
          "<b>转了反而跑不起来</b>。这类保护在代码里通常有注释，读一眼能省很多时间。</font></font>",
          "box", "fail")

    y5 = y4 + 156
    bridge(p, y5, "本页交接：回退机制清楚了。下一步（⑧）把本仓用到的算子<b>列成一张表</b>——"
                  "这张表是这套图最实用的部分。")

    y6 = y5 + 76
    anchors(p, y6, [
        "闸门①：vllm-ascend/vllm_ascend/utils.py:544-603（enable_custom_op）；"
        "失败日志 :590/:597",
        "闸门②：device/hardware_profile.py（能力枚举）；使用示例 ops/rotary_embedding.py:712-719、"
        "sample/sampler.py:163-167",
        "闸门③：配置开关示例 vllm-ascend/vllm_ascend/ascend_config.py:575"
        "（multistream_overlap_shared_expert 默认关）、:615（weight_nz_mode 默认 1）",
        "闸门④⑤：ops/linear_op.py:164-181（unsupported_reason 逐条）、:229-260（_can_fuse）",
        "闸门⑥ 的反面：ops/linear.py:74-79（_should_keep_nd_for_compatibility_weight，"
        "极小形状保持普通布局）、:118-121（原因注释）",
        "三档归一化的具体位置：ops/layernorm.py:92（自研融合）、:118（厂商融合）、"
        ":97（基础）；图级 compilation/passes/norm_quant_fusion_pass.py:385-393",
        "回退相关日志：utils.py:590/597、qknorm_rope_fusion_pass.py:304/310/315/334/341/365、"
        "norm_quant_fusion_pass.py:371/375/398",
    ], h=160)

    y7 = y6 + 194
    concl(p, y7,
          "<b>一句话记住这一页：</b>融合算子要过六道闸门（算子库 / 硬件 / 开关 / 形状 / 布局 / 路径），"
          "任一不过就降档；<b>排障时先找「哪道闸门关了」——日志里 usually 有痕迹，"
          "而「为兼容故意不融合」是最容易被误判成 bug 的一类。</b>")

    legend(p, y7 + 92)
