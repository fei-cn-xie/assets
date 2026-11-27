"""⑧ 算子总表：本仓实际用到的 NPU 算子。"""
from common import head, concl, anchors, legend
from dg import Doc, table

TITLE = "⑧ 算子总表：本仓到底调了哪些 NPU 算子"
SUB = ("用途：这是一个<b>速查表</b>。看到日志/代码里的算子名，回到这张表查「它是干什么的、在哪一类里」。"
       "（≠ 全量清单，只列与推理主链路相关的那些。）")


def build(doc: Doc) -> None:
    p = doc.page("⑧ 算子总表", pw=1560, ph=1640)
    head(p, TITLE, SUB)
    X = 36

    y = 142
    p.text(X, y - 28, 1400, 22,
           "<b>第一组：矩阵乘（占绝大部分计算量）</b>", "text_b")
    rows1 = [
        ["量化矩阵乘（dense）", "W8A8/W4A8/W4A4 的线性层，反量化在算子内", "③ 页"],
        ["批矩阵乘（转置）", "MLA 的输出投影等需要转置的矩阵乘", "⑨ 页"],
        ["分组矩阵乘（MoE）", "一次调用为每个专家算一段 GEMM", "MoE 套 ⑧"],
        ["分组矩阵乘 + SwiGLU + 量化（融合）", "MoE 的两段 GEMM 与激活、量化合成一个算子", "MoE 套 ⑧"],
        ["普通线性", "bf16 权重（NZ 布局）的最基础矩阵乘", "③ 页"],
    ]
    table(p, X, y, [("算子（用途写法）", 520), ("它在算什么", 700), ("详解", 164)], rows1,
          row_h=30)

    y2 = y + 30 * 6 + 22
    p.text(X, y2 - 28, 1400, 22,
           "<b>第二组：归一化 / 激活 / 位置编码（每层跑好几次）</b>", "text_b")
    rows2 = [
        ["基础归一化", "RMSNorm", "④ 页"],
        ["相加 + 归一化 + bias（融合）", "残差相加与归一化一次完成", "④ 页"],
        ["归一化 + 动态量化（融合）", "归一化后直接产出量化张量", "④ 页"],
        ["SwiGLU / clamp 版 / GELU 变体", "门控激活的不同形态", "④ 页"],
        ["激活 + 反量化 + 再量化（融合）", "量化 MLP 的中间段", "④ 页"],
        ["分组激活 + 量化（融合）", "MoE 每组专家的激活", "MoE 套 ⑧"],
        ["通用旋转位置编码", "GQA 模型 prefill/decode 共用", "④ 页"],
        ["归一化 + 旋转 + 写 KV（融合）", "MLA decode 的三合一", "④ 页"],
        ["交错旋转 / 原地部分旋转", "MLA prefill / 稀疏 MLA", "④ 页"],
    ]
    table(p, X, y2, [("算子（用途写法）", 520), ("它在算什么", 700), ("详解", 164)], rows2,
          row_h=30)

    y3 = y2 + 30 * 10 + 22
    p.text(X, y3 - 28, 1400, 22,
           "<b>第三组：attention 与 KV（本套 ⑨ 页专门讲）</b>", "text_b")
    rows3 = [
        ["融合推理 attention（prefill 主用）", "变长拼接布局的 causal attention", "⑨ 页"],
        ["paged attention（decode 可选）", "按块表寻址的 decode attention", "⑨ 页"],
        ["attention 结果归并", "多段/chunk 的 softmax 结果合并", "⑨ 页"],
        ["KV 写入 / 读出（paged）", "按块表把 K/V 写进块缓存", "⑨ 页"],
        ["稀疏 flash attention", "只对 top-k 的 KV 做 attention", "⑨ 页"],
        ["lightning indexer", "为稀疏 attention 选 top-k KV", "⑨ 页"],
        ["MLA prolog / 预处理（融合）", "MLA 的 W_UK/W_UV 预处理段", "⑨ 页"],
    ]
    table(p, X, y3, [("算子（用途写法）", 520), ("它在算什么", 700), ("详解", 164)], rows3,
          row_h=30)

    y4 = y3 + 30 * 8 + 22
    p.text(X, y4 - 28, 1400, 22,
           "<b>第四组：MoE 路由与通信 / 采样</b>", "text_b")
    rows4 = [
        ["gating top-k（融合）", "门卫打分 + 取 top-k 一次完成", "MoE 套 ⑧"],
        ["token 路由 / 重排 / 散射", "把 token 按专家聚起来、算完还原", "MoE 套 ⑧"],
        ["EP dispatch / combine（融合）", "跨卡送 token 与收回（MC2）", "MoE 套 ⑥"],
        ["GEMM + reduce-scatter（融合）", "TP 的通信与计算重叠（MC2）", "⑥ 页"],
        ["集合通信（all-gather / reduce-scatter / all-to-all / all-reduce）",
         "框架层统一接口，底层走 HCCL", "⑥ 页"],
        ["格式转换（ND ↔ NZ）", "加载时把权重转成分形布局", "③ 页"],
    ]
    table(p, X, y4, [("算子（用途写法）", 520), ("它在算什么", 700), ("详解", 164)], rows4,
          row_h=30)

    y5 = y4 + 30 * 7 + 24
    p.box(X, y5, 1424, 100,
          "<b>怎么用这张表</b><br>"
          "<font style=\"font-size:10px\">"
          "① 看到日志里的算子名 → 回到表里查「它属于哪一组」，就知道该去看哪一页；<br>"
          "② 想优化某个环节 → 先确认这一步是「一个融合算子」还是「好几个基础算子拼的」；<br>"
          "③ 想换硬件 → 对照表里的算子逐条确认新平台是否都有，缺哪条就是性能缺口。</font>",
          "box", "cloudp")

    y6 = y5 + 140
    anchors(p, y6, [
        "算子统一封装入口（`torch_npu.*` 与自研 `_C_ascend.*` 都在这里包了一层）："
        "vllm-ascend/vllm_ascend/device/device_op.py（按用途分段，便于查）",
        "自研算子的 C++ 注册与 kernel 目录：vllm-ascend/csrc/torch_binding.cpp"
        "（`TORCH_LIBRARY_EXPAND`）、vllm-ascend/csrc/（各 kernel 子目录）",
        "算子不可用时的占位实现：vllm-ascend/vllm_ascend/ops/__init__.py:37-52（register_dummy_fusion_op）",
        "各组的详细锚点：本套 ③④⑥⑨ 页 + MoE 套 ⑥⑧ 页的页面锚点块",
    ], h=112)

    y7 = y6 + 146
    concl(p, y7,
          "<b>一句话记住这一页：</b>本仓的算子可以归成四组——矩阵乘、小算子（归一化/激活/位置编码）、"
          "attention 与 KV、路由与通信；<b>排查性能时先判断「这一步是融合的还是拼的」。</b>")

    legend(p, y7 + 92)
