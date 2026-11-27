"""⑦ 联动五：enforce-eager 有两个作用域，别混。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc, table

TITLE = "⑦ 联动五：两个 enforce-eager，一个管 target 一个管 draft"
SUB = ("一句话：命令行 <font style=\"font-size:9px\">--enforce-eager</font> 关的是"
       "<b>整个 target 模型的编译与图</b>；"
       "投机配置里的 <font style=\"font-size:9px\">enforce_eager</font> 关的是"
       "<b>只有 draft 那个小头</b>——两者互不校验，也互不告警。")


def build(doc: Doc) -> None:
    p = doc.page("⑦ 两个 enforce-eager", pw=1560, ph=1460)
    head(p, TITLE, SUB)
    X = 36

    # --- 对照表 -----------------------------------------------------------
    y = 150
    p.text(X, y - 32, 1300, 22,
           "<b>第一步：逐项对照（看这三行就够）</b>", "text_b")
    rows = [
        ["写在哪", "--enforce-eager（命令行）", "speculative-config 的 enforce_eager 字段"],
        ["影响谁", "target：编译模式与图模式全置 NONE，capture sizes 清空",
         "只影响 draft：drafter 不用图，走 eager"],
        ["会不会传给对方", "会——构造 draft 时继承 target 的 enforce_eager", "不会回传给 target"],
        ["同时给的后果", "target 已全局 NONE → draft 的这个字段变成<b>冗余无效</b>",
         "无校验、无告警，静默生效"],
    ]
    table(p, X, y, [("", 220), ("--enforce-eager", 620), ("spec.enforce_eager", 580)],
          rows, row_h=40)

    # --- 四种组合 ---------------------------------------------------------
    y2 = y + 40 * 5 + 40
    p.text(X, y2 - 32, 1300, 22,
           "<b>第二步：四种组合的实际效果（本仓用的是第 4 种）</b>", "text_b")
    combos = [
        ("target eager + draft eager", "全 eager：启动最快、稳态最慢", "调试用", "fail"),
        ("target eager + draft 不 eager", "实际仍是全 eager（target 已全局 NONE）",
         "无意义组合", "neutral"),
        ("target 用图 + draft 用图", "最快，但形状/后端限制多", "保守场景可用", "cloudp"),
        ("target 用图 + draft eager", "decode 侧推荐：图赢了提交开销，draft 反正很小",
         "本仓 D 侧", "cloudd"),
    ]
    cw = 340
    for i, (name, effect, note, role) in enumerate(combos):
        cx = X + i * (cw + 12)
        p.box(cx, y2, cw, 132,
              f"<b>{name}</b><br>"
              f"<font style=\"font-size:10px\">{effect}</font><br>"
              f"<font style=\"font-size:9px;color:#666666\">{note}</font>",
              "box", role)

    # --- 本仓实例 ---------------------------------------------------------
    y3 = y2 + 172
    p.box(X, y3, 1424, 116,
          "<b>第三步：本仓的实例（一眼看懂「为什么会这么配」）</b><br>"
          "<font style=\"font-size:10px\">"
          "· <b>D 侧</b>：不加 <font style=\"font-size:9px\">--enforce-eager</font>，"
          "并给 <font style=\"font-size:9px\">FULL_DECODE_ONLY</font>；"
          "同时 speculative-config 里写 <font style=\"font-size:9px\">enforce_eager: true</font> "
          "→ <b>target 跑图、draft 跑 eager</b>。<br>"
          "· <b>P 侧与 edge</b>：直接加 <font style=\"font-size:9px\">--enforce-eager</font>"
          "（P 只跑一次就交接，省启动时间更划算；edge 的 forward 本来就被劫持）。<br>"
          "· 有一个包（mooncake_pd_195）的 decode 侧<b>没有</b> enforce-eager，还手写了 capture_sizes。</font>",
          "box", "data")

    # --- 一个坑 -----------------------------------------------------------
    y4 = y3 + 156
    panel(p, X, y4, 1424, 140, "一个容易踩的坑：以为「draft eager」就不影响 target 的图", [
        "反过来的方向才会互相牵连：target 一旦 eager，全局图模式就是 NONE，draft 想用图也用不了。",
        "而 draft eager 只是「draft 自己不用图」，target 的图照常工作——这正是本仓 D 侧的选择。",
        "另外：某些模型族（例如 GLM 系列）会被平台<b>强制</b>把 draft 切成 eager 并打 warning，"
        "那时你的配置会被覆盖，别以为是自己的参数写错了。",
    ], "fail")

    y5 = y4 + 180
    bridge(p, y5, "本页交接：两个 eager 分清了。下一步（⑧）看「引擎在什么情况下会主动给你降级」——"
                  "降级链是这条联动脉络里唯一会打日志的部分。")

    y6 = y5 + 76
    anchors(p, y6, [
        "target 侧：vllm/config/model.py:242-249；生效点 vllm/config/vllm.py:1591-1597、"
        ":1823-1829（capture sizes 清零）",
        "draft 侧字段：vllm/config/speculative.py:378-379；只有三处读它："
        "vllm/v1/worker/gpu_model_runner.py:6214、vllm/v1/spec_decode/llm_base_proposer.py:430-439、"
        "extract_hidden_states.py:249",
        "target → draft 的继承：vllm/config/speculative.py:1289",
        "昇腾 xlite 回填差异：vllm-ascend/vllm_ascend/platform.py:1208-1211",
        "GLM 强制 draft eager：vllm-ascend/.../spec_decode/llm_base_proposer.py:271-290",
        "本仓实例：assets/c_home/bash/pds/30rc_1124/sidecar/cloud_d.sh:30-31（"
        "spec enforce_eager + FULL_DECODE_ONLY）与 cloud_p.sh:26（--enforce-eager）",
    ], h=128)

    y7 = y6 + 162
    concl(p, y7,
          "<b>一句话记住这一页：</b>命令行那个管整个 target，JSON 里那个只管 draft；"
          "「target 用图 + draft eager」是 D 侧的常规选择，"
          "<b>但 target 一旦 eager，draft 想用图也用不了。</b>")

    legend(p, y7 + 92, extra_fill=("draft/投机侧", "cloudd"))
