"""⑧ MTP 的训练期与推理期：同一个头，两种命运。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc

TITLE = "⑧ MTP 的头：训练期学「往后看几步」，推理期当「便宜的起草者」"
SUB = ("一句话：MTP 头在训练时被一个<b>额外的预测损失</b>训练出来；"
       "训练完之后，它的用途变成<b>推理时的 draft</b>。"
       "所以「MTP」既是训练概念，也是推理概念——本页把两半接起来。")


def build(doc: Doc) -> None:
    p = doc.page("⑧ MTP 训练与推理", pw=1560, ph=1500)
    head(p, TITLE, SUB)
    X = 36

    # --- 结构（本仓事实） -------------------------------------------------
    y = 150
    p.text(X, y - 32, 1300, 22,
           "<b>第一步：MTP 层的结构（这些<b>全都能在仓里查到</b>）</b>", "text_b")
    p.box(X, y, 1424, 128,
          "<font style=\"font-size:10px\">"
          "<b>输入</b>：① 上一层（或上一个深度）的 hidden states；② 下一个位置的 token embedding<br>"
          "<b>处理</b>：两份各自过一层归一化 → 拼接（维度翻倍）→ 一个线性层压回 hidden 维度<br>"
          "<b>输出</b>：走<b>共享的 lm_head</b>（与主模型同一个输出头）得到词表分布<br>"
          "<font color=\"#c0392b\">关键点：它<b>不复制</b> embedding 与 lm_head，而是复用主模型的——"
          "所以额外参数很少（这是「MTP 省显存」的根本原因）。</font></font>",
          "box", "cloudd")

    # --- 两半 -------------------------------------------------------------
    y2 = y + 168
    p.text(X, y2 - 32, 1300, 22,
           "<b>第二步：同一个头，训练与推理两个用途</b>", "text_b")
    panel(p, X, y2, 700, 190, "训练期：多 token 预测（多一个损失）", [
        "主模型照常预测下一个词（第 1 个）；MTP 头被训练去预测「再往后一个」（第 2 个）。",
        "所以它的损失是「用它算的 log 概率，对正确答案取负」，再乘一个权重加到总损失上。",
        "这个权重字段就在 config 里，注释明写：<b>训练期用、推理期不用</b>。",
        "<font color=\"#c0392b\">（具体的目标函数形式与常见权重取值属通用知识；"
        "仓内只证明「存在这样一个权重字段」，不实现这个损失。）</font>",
    ], "cloudp")
    panel(p, X + 760, y2, 664, 190, "推理期：当 draft（投机解码的起草者）", [
        "权重冻结后，MTP 头就变成一个「便宜的下一步预测器」：",
        "吃主模型的 hidden，猜出接下来 K 个 token；主模型再一次前向验证。",
        "此时它<b>不再产生损失</b>，只产生候选。",
        "→ 这就是 MTP 套 ⑧ 讲的「draft 走 eager、target 走图」。",
    ], "cloudd")

    # --- checkpoint 键名 --------------------------------------------------
    y3 = y2 + 230
    p.text(X, y3 - 32, 1300, 22,
           "<b>第三步：一个很实用的细节——MTP 头在权重文件里长什么样</b>", "text_b")
    p.box(X, y3, 1424, 118,
          "<font style=\"font-size:10px\">"
          "· 有的 checkpoint 把它写成独立的 <b>mtp.{i}.*</b> 前缀；"
          "有的写成 <b>model.layers.{总层数+i}.*</b>（假装它是主干之后的第几层）。<br>"
          "· 加载器因此要做两件事：<b>①</b> 把 `mtp.` 重映射到「主干层号之后」；"
          "<b>②</b> 让<b>主模型</b>在加载时主动<b>跳过</b>这些层（否则会报「意外的权重」）。<br>"
          "· 仓内把「映射到 None 就表示丢弃这个键」作为通用机制——"
          "所以主模型对 `mtp.` 的处理就是一句「前缀→None」。<br>"
          "<font color=\"#c0392b\">这也解释了为什么有些模型目录里「多出来一堆 mtp 权重」却能正常加载。</font></font>",
          "box", "neutral")

    # --- 一个数字 ---------------------------------------------------------
    y4 = y3 + 158
    p.box(X, y4, 1424, 96,
          "<b>为什么「训练期的损失权重」在推理侧完全用不到</b><br>"
          "<font style=\"font-size:10px\">"
          "那个权重只决定「训练时 MTP 的误差对总梯度的贡献有多大」，"
          "而推理时不更新任何参数 → 与它无关。"
          "它在 config 里被保留下来，只是因为 config.json 是<b>一份跨阶段共享的文件</b>："
          "训练写进去，推理读进来但忽略。<font color=\"#c0392b\">"
          "这也是「训练期字段被带进推理配置」的典型例子。</font></font>",
          "box", "data")

    y5 = y4 + 136
    bridge(p, y5, "本页交接：MTP 的两半接上了。下一步（⑨）回到加载器视角——"
                  "看它如何在同一个目录里只挑它要的文件。")

    y6 = y5 + 76
    anchors(p, y6, [
        "MTP 层结构（enorm/hnorm/eh_proj/共享 lm_head）："
        "vllm/model_executor/models/deepseek_mtp.py:80-124、:136-147",
        "「训练期字段、推理不用」原文：vllm/transformers_utils/configs/hy_v4.py:53-54；"
        "默认值 :130-131",
        "另一处同类字段（MoE 的辅助损失开关）：vllm/transformers_utils/configs/hy_v3.py:84-87",
        "checkpoint 键名重映射 mtp.{i}.* → model.layers.{N+i}.*："
        "vllm/models/deepseek_v4/nvidia/mtp.py:392-398；"
        "识别与跳过 vllm/model_executor/models/utils.py:543-568、:571-587",
        "主模型对 mtp. 前缀的丢弃：vllm/model_executor/models/qwen3_next.py:810 等（"
        "WeightsMapper(orig_to_new_prefix={'mtp.': None})）",
        "WeightsMapper 的「映射成 None 即忽略」机制：vllm/model_executor/models/utils.py:46-57",
    ], h=136)

    y7 = y6 + 170
    concl(p, y7,
          "<b>一句话记住这一页：</b>MTP 头 = 训练期多学一个「往后看」的目标、推理期当便宜的起草者；"
          "它的损失权重只影响训练，而它的权重在文件里可能写成 `mtp.*` 或「主干之后的层」——"
          "<b>加载器靠「映射成 None 就丢弃」把两半分开。</b>")

    legend(p, y7 + 92, extra_fill=("训练期语义（多为通用知识）", "cloudp"))
