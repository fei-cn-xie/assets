"""⑧ 回放期：一次 execute_model 里图在哪、图外还剩什么。"""
from common import head, concl, anchors, legend
from dg import Doc

TITLE = "⑧ 回放期：一次 decode step 从 scheduler 到 replay 再到吐 token"
SUB = ("一句话：<b>图只覆盖「模型 forward」这一段</b>；"
       "进图前要备好静态输入，出图后还要算 logits、采样、回收——这些都在图外。")


def row(p, y, no, title, body, role, h=62):
    p.box(36, y, 1424, h,
          f"<b>{no}　{title}</b><br>"
          f"<font style=\"font-size:9px\" color=\"#333333\">{body}</font>",
          "box", role)
    return y + h + 10


def build(doc: Doc) -> None:
    p = doc.page("⑧ 回放路径", pw=1560, ph=1420)
    head(p, TITLE, SUB)
    X = 36

    y = 156
    p.text(X, y - 32, 1300, 22,
           "<b>把一次 step 从头走到尾，标出「图内 / 图外」</b>", "text_b")

    y = row(p, y, "①", "调度器出计划",
            "SchedulerOutput：每请求算几个 token + 新块的 block_ids。"
            "<font color=\"#666666\">（图外，EngineCore 主线程）</font>", "neutral")
    y = row(p, y, "②", "决定这一步用哪个桶 / 哪张图",
            "按 padded num_tokens + uniform + LoRA 数查表 → 得到 "
            "cudagraph_mode 与 batch_descriptor。<b>查不到就是 NONE（eager）</b>。", "cloudp")
    y = row(p, y, "③", "备好静态输入（图外）",
            "把本步的 input_ids / positions / slot_mapping / seq_lens 用 copy_ "
            "写进持久 buffer 的前 N 个位置；slot_mapping 按 padded 长度补齐。", "cloudd")
    y = row(p, y, "④", "进 forward context（图外）",
            "set_forward_context(attn_metadata, ..., num_tokens=num_tokens_padded, "
            "cudagraph_runtime_mode=<mode>, batch_descriptor=<desc>)——"
            "把「本步用哪张图」写进上下文，wrapper 在 forward 里读它。", "cloudd")
    y = row(p, y, "⑤", "命中图 → replay()（图内只这一下）",
            "显存里已有这张图 → 一条 <b>graph.replay()</b>；CPU 只提交一次，"
            "设备按录好的顺序跑完整个 forward。", "cloudp", h=68)
    y = row(p, y, "⑥", "取输出（图外，且输出是「池子里那块静态显存」）",
            "写 right 的是捕获期就定好的输出 buffer（弱引用）；"
            "再用它算 logits——注意 logits 与采样<b>不在图里</b>。", "cloudd")
    y = row(p, y, "⑦", "采样 + 回收 + 回 CPU（图外）",
            "选 token、判结束、把结果搬回 CPU、写进 EngineCoreOutputs 发给前端。", "fail")
    y = row(p, y, "⑧", "KV 真正写进去的时机",
            "写 KV 发生在 attention 自己的 op 里（图内），"
            "但<b>写到哪</b>由进图前算好的 slot_mapping 决定；"
            "padding 的位置填 -1，等于跳过。", "data")

    y2 = y + 6
    p.box(X, y2, 1424, 108,
          "<b>为什么「采样 / logits」一定要留在图外（三个理由）</b><br>"
          "<font style=\"font-size:10px\">"
          "① <b>形状不定</b>：这一步要采几个 token 取决于 batch 里各请求的状态，"
          "而且 spec decode 还要按 draft 结果变；<br>"
          "② <b>要问 CPU</b>：判结束、算 stop 条件、生成输出事件都是 CPU 逻辑，"
          "进了图就再也问不了；<br>"
          "③ <b>要回主机</b>：把 sampled token 搬回 CPU 是同步操作，图里禁止。</font>",
          "box", "neutral")

    y3 = y2 + 142
    anchors(p, y3, [
        "execute_model 与选路：vllm/v1/worker/gpu_model_runner.py:4188、:4289-4310、:3954-4066",
        "进 forward context：同文件 :4446-4457；命中后 replay：vllm/compilation/cuda_graph.py:242-246、"
        ":258-265、:362",
        "V2 的显式 replay（FULL 连输入都不传）：vllm/v1/worker/gpu/model_runner.py:1901-1909；"
        "vllm/v1/worker/gpu/cudagraph_utils.py:529-542",
        "图外算 logits / 采样：gpu_model_runner.py:4478-4499（logits）、:4567（sample_tokens）、"
        ":3690（回 CPU 记账）",
        "KV 写入与 slot_mapping：:4388-4397、:4455；"
        "vllm/model_executor/layers/attention/attention.py:750-794",
    ], h=112)

    y4 = y3 + 146
    concl(p, y4,
          "<b>一句话记住这一页：</b>图只包住「模型 forward」；"
          "进图前备好静态输入并告诉上下文用哪张图，出图后算 logits / 采样 / 回 CPU。"
          "所以图能省的只是<b>forward 内部的下发次数</b>，省不了采样与调度。")

    legend(p, y4 + 92, extra_fill=("数据面（KV 写入）", "data"))
