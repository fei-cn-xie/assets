"""⑯ 权重怎么参与计算：一次 forward 的调用链（加载的终点就是计算的起点）。"""
from common import head, concl, anchors, legend
from dg import Doc, table

TITLE = "⑯ 权重怎么参与计算：一次 forward 的调用链"
SUB = ("一句话：<b>加载时写的是 layer.weight，计算时读的也是同一个 layer.weight</b>"
       "——中间没有第二次搬运。所谓「参与计算」，就是把激活张量和这份权重做矩阵乘。")
X = 36


def build(doc: Doc) -> None:
    p = doc.page("⑯ 权重参与计算", pw=1560, ph=1440)
    head(p, TITLE, SUB)

    # --- 左：调用链 -------------------------------------------------------
    y = 150
    p.text(X, y - 32, 700, 22,
           "<b>一条调用链走到底（dense 模型，例如 Qwen/Llama 类）</b>", "text_b")
    chain = [
        ("model.forward(input_ids)", "入口", "neutral"),
        ("embed_tokens(ids) → 查表拿词向量",
         "用的是 embedding.weight（TP 时按词表切）", "cloudd"),
        ("× N 层：layernorm → attention → layernorm → MLP", "逐层重复", "neutral"),
        ("qkv_proj(hidden_states) → GEMM", "qkv_proj.weight（列并行，切输出维）", "cloudp"),
        ("attention(q,k,v) → 读写 KV cache", "KV cache 不是权重，是运行期数据", "data"),
        ("o_proj → 行并行 GEMM + all_reduce", "o_proj.weight（切输入维）", "cloudp"),
        ("gate_up_proj → SiLU → down_proj + all_reduce",
         "MLP 权重（同样列/行并行）", "cloudp"),
        ("lm_head(hidden) → logits", "lm_head.weight（按词表切）", "cloudd"),
    ]
    cy = y
    for i, (title, note, role) in enumerate(chain):
        p.box(X, cy, 700, 56,
              f"<b>{title}</b><br>"
              f"<font style=\"font-size:9px\" color=\"#666666\">{note}</font>",
              "box", role)
        cy += 66

    # --- 右：三处"用权重"的细节 -------------------------------------------
    p.text(X + 760, y - 32, 700, 22,
           "<b>右侧：三处细节（TP 时各自怎么切、谁做通信）</b>", "text_b")
    p.box(X + 760, y, 664, 128,
          "<b>① 列并行 Linear（qkv_proj / gate_up_proj）</b><br>"
          "<font style=\"font-size:10px\">"
          "权重按<b>输出维</b>切：每个 rank 只算自己那部分输出（比如自己那 1/4 个 head "
          "或 1/4 的中间维）。<br>"
          "特点：forward 里<b>不需要通信</b>——算完各 rank 手上是一段不同的特征。</font>",
          "box", "cloudp")
    p.box(X + 760, y + 148, 664, 128,
          "<b>② 行并行 Linear（o_proj / down_proj）</b><br>"
          "<font style=\"font-size:10px\">"
          "权重按<b>输入维</b>切：每个 rank 都在自己那段输入上做乘加，"
          "得到的是<b>部分和</b>。<br>"
          "特点：forward 末尾<b>必须 all_reduce</b> 把 4 个 part 加起来——"
          "这是 TP 的通信成本来源。</font>",
          "box", "cloudp")
    p.box(X + 760, y + 296, 664, 128,
          "<b>③ 并行 Embedding / LMHead</b><br>"
          "<font style=\"font-size:10px\">"
          "按<b>词表维</b>切：每个 rank 只持有词表的一段（尾部要补零对齐）。<br>"
          "查表时先看这个 id 在不在自己那段，不在就返回 0，最后 all_reduce；"
          "lm_head 反过来按词表段算 logits。</font>",
          "box", "cloudd")

    # --- 权重 vs KV cache -------------------------------------------------
    y2 = y + 8 * 66 + 30
    p.text(X, y2 - 32, 1300, 22,
           "<b>最容易混的一件事：权重和 KV cache 都被「加载」，但不是一回事</b>",
           "text_b")
    rows = [
        ["从哪里来", "磁盘上的 safetensors 文件", "运行时算出来的（没有文件）"],
        ["什么时候进显存", "启动时（加载阶段）", "每次请求跑到时按块分配"],
        ["大小由什么决定", "模型结构与精度（固定）", "并发请求数 × 上下文长度（会变）"],
        ["会被写吗", "加载后只读（除格式重排）", "每个 token 都在写新的块"],
        ["不够了怎么办", "起不来：降 gpu-memory-utilization / 换小模型", "抢占（丢 KV 重算）"],
    ]
    table(p, X, y2, [("", 200), ("权重（weights）", 560), ("KV cache", 560)], rows,
          row_h=34)

    y3 = y2 + 34 * 6 + 34
    p.box(X, y3, 1424, 92,
          "<b>所以「加载完就能算」这句话的准确含义</b><br>"
          "<font style=\"font-size:10px\">"
          "① 权重已经在 NPU 上（layer.weight 可以直接参与 GEMM）；"
          "② KV 块池也已切好（有地方放历史 K/V）；"
          "③ 两者的边界是 <b>profile</b>：权重占多少 → 激活要多少 → 剩下的都给 KV。<br>"
          "这也解释了为什么「加大模型的上下文」不会变慢、却会<b>减少并发</b>——"
          "上下文长了，同一个 KV 池能装下的请求数就少了。</font>",
          "box", "neutral")

    y4 = y3 + 126
    anchors(p, y4, [
        "forward 主干：vllm/model_executor/models/qwen3.py:318（LM）、"
        "qwen2.py:401（Model）、:226（DecoderLayer）、:153/158（Attention 的 qkv_proj）",
        "Linear 落地：vllm/model_executor/layers/linear.py:408-414（LinearBase.forward）、"
        ":228-238（UnquantizedLinearMethod.apply）、"
        "vllm/model_executor/layers/utils.py:84-90（= F.linear）",
        "列/行并行的切分与通信：linear.py:572-606（列）、:1730-1776（行 + all_reduce）",
        "Embedding 查表：vllm/model_executor/layers/vocab_parallel_embedding.py:510-534",
        "昇腾算子：vllm_ascend/ops/linear.py:133-139（unquantized_gemm）、"
        "vllm_ascend/attention/attention_v1.py:1248-1271（paged attention 入口）",
    ], h=104)

    y5 = y4 + 136
    concl(p, y5,
          "<b>一句话记住这一页：</b>加载的终点就是计算的起点——"
          "`param.data.copy_` 写进去的 `layer.weight`，在 forward 里被 GEMM 直接读；"
          "TP 时「列并行不通信、行并行末尾 all_reduce」，这就是 4 张卡协同算一层的方式。")

    legend(p, y5 + 92, extra_fill=("数据面（KV，不是权重）", "data"))
