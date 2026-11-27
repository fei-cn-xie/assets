"""⑤ 激活：训练的「账本」 vs 推理的「峰值」。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc

TITLE = "⑤ 激活：训练要留「账本」，推理只要「峰值」"
SUB = ("一句话：每一层的中间结果（激活）在训练时要<b>留到反向他用</b>，"
       "在推理时<b>算完立刻可以扔</b>——所以推理只关心「同时活着的那一批有多大」，"
       "这个量必须实测（profile），而它随 token 数线性膨胀。")


def build(doc: Doc) -> None:
    p = doc.page("⑤ 激活与 workspace", pw=1560, ph=1480)
    head(p, TITLE, SUB)
    X = 36

    # --- 两种模式对照 -----------------------------------------------------
    y = 150
    p.text(X, y - 32, 1300, 22,
           "<b>第一步：同一层、同一个中间结果，两种命运</b>", "text_b")
    panel(p, X, y, 700, 168, "训练：留下来（账本）", [
        "反向传播要用「这个算子的输入」去算梯度，",
        "所以 autograd 默认把前向的中间结果全部驻留。",
        "层数 × batch × 序列长度 × 隐藏维度 → 很容易成为显存第一大户。",
        "缓解手段：重算（checkpointing）、FlashAttention 式的重算注意力。",
    ], "cloudp")
    panel(p, X + 760, y, 664, 168, "推理：算完就扔（峰值）", [
        "没有反向 → 中间结果用完立刻可释放；",
        "所以只需要知道「某一时刻最多同时有多少」= 峰值。",
        "这个峰值是<b>实测</b>出来的：启动时用最大 batch 空跑一遍。",
        "然后才能算「还剩多少显存给 KV」。",
    ], "cloudd")

    # --- 一个真实算术 -----------------------------------------------------
    y2 = y + 208
    p.text(X, y2 - 32, 1300, 22,
           "<b>第二步：为什么峰值不能忽略（一个从 288 MB 涨到 6 GB 的真实算术）</b>",
           "text_b")
    p.box(X, y2, 1424, 128,
          "<font style=\"font-size:10px\">"
          "昇腾 attention 的 chunked-prefill workspace 是「按 token 上限 × 头维度」分配的。"
          "仓内注释给了两个数：<br>"
          "· 按 576 的头维度、fp16 算 → 约 <b>288 MB</b>；<br>"
          "· 换成 192 的 QK 头维度 × 128 个头 → 同一套公式就变成 <b>约 6 GB</b>。<br>"
          "<font color=\"#c0392b\">同一份代码、同一个 token 上限，只因为头配置不同就差 20 倍——"
          "所以「激活峰值」必须实测，不能按经验拍。</font>"
          "仓内另有一处注释直接写「workspace 显存随这个 token 上限线性增长」。</font>",
          "box", "fail")

    # --- 三个影响 ---------------------------------------------------------
    y3 = y2 + 168
    p.text(X, y3 - 32, 1300, 22,
           "<b>第三步：由「峰值」推出的三个实践结论</b>", "text_b")
    panel(p, X, y3, 460, 156, "① profile_run 存在的唯一理由", [
        "启动时用「最大 batch」空跑一遍前向，",
        "记录这一刻的显存峰值，再从可用显存里扣掉。",
        "没有这一步，KV 预算就只能猜。",
    ], "cloudd")
    panel(p, X + 482, y3, 460, 156, "② 为什么调大 batched tokens 会挤掉 KV", [
        "一步能装的 token 越多，workspace / 激活峰值越大，",
        "扣掉之后留给 KV 的就越少。",
        "所以「吞吐」与「并发」并不总能同时调高。",
    ], "fail")
    panel(p, X + 964, y3, 460, 156, "③ 为什么 MoE 还要两块 scratch", [
        "MoE 前向需要一块放 GEMM 输出、一块放激活函数的中间结果，",
        "形状按「token 数 × max(N, K)」分配。",
        "这也算在峰值里，且随 token 数线性涨。",
    ], "neutral")

    # --- 一句话 -----------------------------------------------------------
    y4 = y3 + 196
    p.box(X, y4, 1424, 96,
          "<b>一句话把两边分开</b><br>"
          "<font style=\"font-size:10px\">"
          "训练：<b>激活 = 账本</b>，为了算梯度必须留着，显存不够就重算或切分。<br>"
          "推理：<b>激活 = 峰值</b>，只要知道最大值并留出余量，剩下的显存全给 KV cache。<br>"
          "所以「推理为什么能开很多并发」的答案之一就藏在这句话里。</font>",
          "box", "cloudd")

    y5 = y4 + 136
    bridge(p, y5, "本页交接：激活讲完，显存四项就齐了。下一步（⑥）讲精度——"
                  "为什么推理能整份用低精度，而训练必须留一份高精度。")

    y6 = y5 + 76
    anchors(p, y6, [
        "推理只量峰值：vllm/v1/worker/gpu_worker.py:850-854（non_kv = 权重+非torch+激活峰值+图）；"
        "算例 vllm/utils/mem_utils.py:242-273",
        "workspace 算术（288 MB → 6 GB）："
        "vllm-ascend/vllm_ascend/attention/utils.py:256-281（注释 :261-273）",
        "workspace 随 token 上限线性增长：vllm-ascend/vllm_ascend/ascend_config.py:590-597",
        "MoE 两块 scratch：vllm/model_executor/layers/fused_moe/modular_kernel.py:960-963",
        "显存不够先牺牲 workspace 的实例："
        "vllm-ascend/vllm_ascend/_310p/worker_310p.py:130、:150（日志 halved for workspace）",
        "训练侧「留到反向他用」：通用知识（仓内仅有 activation checkpointing 的说明性文档串，"
        "phi4mm_audio.py:114-124）",
    ], h=128)

    y7 = y6 + 162
    concl(p, y7,
          "<b>一句话记住这一页：</b>推理只关心激活的<b>峰值</b>，且这个峰值随 token 上限线性涨；"
          "它先被扣掉、剩下的才是 KV；<b>所以「调大吞吐参数」和「调高并发」经常是互相拆台的。</b>")

    legend(p, y7 + 92, extra_fill=("推理可释放的中间结果", "cloudd"))
