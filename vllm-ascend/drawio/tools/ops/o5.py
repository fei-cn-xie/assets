"""⑤ 反直觉的取舍：采样那一步主动不用硬件融合算子。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc, table

TITLE = "⑤ 采样：为什么「硬件有融合算子」却主动不用它"
SUB = ("一句话：本仓采样路径里有两处**刻意的反向选择**——"
       "① top-k/top-p 不走硬件的融合算子，改走排序+掩码；"
       "② 不用现成的多项式采样，改用自己的「指数噪声 + 取最大」。"
       "两处都是为了避开<b>跨设备同步</b>与<b>固定延迟</b>。")


def build(doc: Doc) -> None:
    p = doc.page("⑤ 采样的取舍", pw=1560, ph=1520)
    head(p, TITLE, SUB)
    X = 36

    # --- 第一处反直觉 -----------------------------------------------------
    y = 148
    p.text(X, y - 30, 1300, 22,
           "<b>第一步：反直觉之一——top-k/top-p 主动放弃硬件算子</b>", "text_b")
    p.box(X, y, 1424, 124,
          "<font style=\"font-size:10px\">"
          "硬件其实提供了「top-k/top-p」的融合算子，但本仓的代码注释把它<b>否掉了</b>，原文大意：<br>"
          "「用基于排序的 PyTorch 实现。<b>硬件的 top-k/top-p 在 k 较大、或批内 k 混杂时会严重劣化（5~28 ms）</b>，"
          "而排序 + 掩码稳定在 ~1 ms。」<br>"
          "<font color=\"#c0392b\">这是个非常典型的取舍：<b>融合算子不一定更快</b>——"
          "它的性能对「形状是否规整」极其敏感，而采样的 batch 恰恰最不规整。</font></font>",
          "box", "fail")

    # --- 第二处反直觉 -----------------------------------------------------
    y2 = y + 164
    p.text(X, y2 - 30, 1300, 22,
           "<b>第二步：反直觉之二——不用现成的多项式采样</b>", "text_b")
    p.box(X, y2, 1424, 132,
          "<font style=\"font-size:10px\">"
          "现成的「按概率抽一个」的算子会在内部<b>把结果读回 CPU</b>（或触发同步）才能继续，"
          "而采样这一步的延迟极度敏感（它在每个 decode step 的尾巴上）。<br>"
          "本仓的做法是：在一张<b>旁路流</b>上生成一组指数分布随机数 → 与概率相除 → 取最大值。"
          "数学上等价于「按概率采样」，但<b>全程留在设备上</b>，不需要把结果读回来。<br>"
          "<font color=\"#1c7c54\">最后再把这条旁路流与主流对齐（wait_stream），"
          "并用 record_stream 告诉分配器这块张量还在被用。</font></font>",
          "box", "cloudd")

    # --- 采样前后的完整链 -------------------------------------------------
    y3 = y2 + 172
    p.text(X, y3 - 30, 1300, 22,
           "<b>第三步：采样前后其实是一串加工（这串里既有融合算子也有通用算子）</b>",
           "text_b")
    rows = [
        ["① logits 提到 fp32", "元素级通用算子", "为了数值稳定"],
        ["② 罚分（重复/存在/频率）", "Triton 实现的小 kernel", "有 Triton 才用"],
        ["③ top-k / top-p 截断", "排序 + 掩码（<b>不用</b>硬件融合版）", "见第一步"],
        ["④ 温度缩放 / softmax", "元素级算子", "得到概率"],
        ["⑤ 采样（随机或贪心）", "贪心 = 取最大；随机 = 指数噪声法", "见第二步"],
        ["⑥ logprobs（可选）", "取 top-k 的通用算子", "返回给调用方"],
    ]
    table(p, X, y3, [("步骤", 320), ("落到什么算子", 620), ("备注", 484)], rows,
          row_h=34)

    # --- 一条通用规律 -----------------------------------------------------
    y4 = y3 + 34 * 7 + 32
    p.box(X, y4, 1424, 116,
          "<b>第四步：由这两处取舍可以抽出一条通用规律</b><br>"
          "<font style=\"font-size:10px\">"
          "<b>「融合算子」适合形状规整、批大、计算密的场景（prefill / MLP / 矩阵乘）；</b><br>"
          "<b>「通用算子 + 旁路流」适合形状不规整、批小、且处在关键延迟路径上的场景（采样、小算子拼装）。</b><br>"
          "<font color=\"#c0392b\">所以「为什么不用那个更快的专用算子」这类问题，"
          "答案往往不是「没做」，而是「做了但在这个形状下更慢」——本仓至少有两处注释明确这么写。</font></font>",
          "box", "neutral")

    y5 = y4 + 156
    bridge(p, y5, "本页交接：采样是「尾部」。下一步（⑥）看「腰部」——"
                  "并行带来的通信算子，以及它怎么和计算重叠。")

    y6 = y5 + 76
    anchors(p, y6, [
        "top-k/top-p 的两条实现与「主动不用硬件」的注释原文："
        "vllm-ascend/vllm_ascend/sample/sampler.py:150-160"
        "（注释：npu_top_k_top_p degrades severely (5-28ms) … sort+mask is consistently ~1ms）；"
        "门控点 :163-167（HardwareCapability）；能力定义 device/hardware_profile.py:140-141",
        "不用现成多项式采样的理由（会引起 CPU-NPU 同步）：sample/sampler.py:24-26",
        "旁路流采样实现：sample/sampler.py:31-42（npu_stream_switch → exponential_ → div → argmax，"
        "wait_stream :40、record_stream :41）",
        "贪心 = 取最大：sample/sampler.py:83-84",
        "罚分走 Triton：sample/sampler.py:52-58（penalties）；logprobs 用 topk："
        "worker/v2/sample/logprob.py:136",
        "MoE 路由里同类注释（torch.topk 在 bf16 图模式下不支持）："
        "ops/fused_moe/router/grouped_topk_router.py:104-107",
    ], h=144)

    y7 = y6 + 178
    concl(p, y7,
          "<b>一句话记住这一页：</b>采样路径有两处「反向选择」——放弃硬件 top-k/top-p（形状不规整时慢 5~28 ms）、"
          "放弃现成多项式采样（会引起跨设备同步）；"
          "<b>规律是：融合算子吃「规整与大批」，通用算子吃「不规整与小批、且延迟敏感」。</b>")

    legend(p, y7 + 92)
