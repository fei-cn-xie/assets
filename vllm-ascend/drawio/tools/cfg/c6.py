"""⑥ 联动四：边云 PD 上的硬约束（块大小、并行、请求改写）。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc

TITLE = "⑥ 联动四：开了 PD 之后，一批参数从「建议」变成「必须」"
SUB = ("一句话：PD 要求 P 与 D 在<b>块大小、并行度、拓扑</b>上对齐，"
       "否则连不上或算错；而且 P 侧收到的请求会被引擎<b>强制改写</b>（max_tokens=1 等）。")


def build(doc: Doc) -> None:
    p = doc.page("⑥ PD 硬约束", pw=1560, ph=1520)
    head(p, TITLE, SUB)
    X = 36

    # --- 五条必须对齐 -----------------------------------------------------
    y = 150
    p.text(X, y - 32, 1300, 22,
           "<b>第一步：五条「两侧必须一致」（不一致会在启动或首个请求时报错）</b>",
           "text_b")
    from dg import table
    rows = [
        ["块大小（hash 粒度）", "两侧 block_size × DCP 必须相等",
         "「PD manifest and cloud hash block sizes differ」"],
        ["P/D 的 TP", "要么相等，要么满足「P ≥ D 且能整除」（否则切片优化静默关闭）",
         "kv_connector_extra_config 里的 dp_size/tp_size 要写部署级真实值"],
        ["拓扑规模", "edge 必须 DP=1；cloud 的 DP 必须等于拓扑里写的 DP 数",
         "引擎启动时校验，不符直接 ValueError"],
        ["LWD 并行白名单", "不支持 PP / PCP / DCP / 弹性 EP /跨 DP 的 embedding",
         "配置期就 fail-fast"],
        ["KV 端口与 rank", "每个 DP/TP 组合的 kv_port 不能撞",
         "用 data_parallel_index 派生，不是 data_parallel_rank"],
    ]
    table(p, X, y, [("项", 260), ("要求", 620), ("踩中时的表现", 540)], rows,
          row_h=40)

    # --- P 侧请求被改写 ---------------------------------------------------
    y2 = y + 40 * 6 + 40
    p.text(X, y2 - 32, 1300, 22,
           "<b>第二步：PD 会<b>强制改写</b>请求字段（用户传的会被覆盖）</b>", "text_b")
    p.box(X, y2, 1424, 104,
          "<font style=\"font-size:10px\">"
          "P 侧（prefill 实例）收到请求后，引擎会把它改成："
          "<b>max_tokens=1、min_tokens=1、ignore_eos=True、抑制用户可见输出</b>，"
          "并设置 KV 传输标记（远端 prefill/decode 各一套）。<br>"
          "D 侧相反：打开「远端 prefill」并<b>跳过本地前缀缓存读取</b>"
          "（因为 KV 是别人给的，读本地缓存会错）。<br>"
          "<font color=\"#c0392b\">含义：P 实例不产出用户可见 token，它只负责把 KV 算出来交出去。</font></font>",
          "box", "cloudp")

    # --- 长度预算 ---------------------------------------------------------
    y3 = y2 + 144
    panel(p, X, y3, 700, 152, "长度预算：PD 各段必须装得下", [
        "P 需要 required = 交接点 + 1 个位置；D 需要 n + m（输入 + 输出）。",
        "超过 max_model_len 直接报「PD stage input/output length exceeds the model budget」。",
        "这就是为什么 PD 包里 max_model_len 要从「最长 prompt + 生成长度」反推。",
    ], "fail")
    panel(p, X + 760, y3, 664, 152, "与 MTP 的叠加（本仓真实约束）", [
        "D 侧一次要算 1+K 个位置 → 尾段长度必须能对齐 1+K；",
        "MTP 层还有自己独立的 KV（作为额外层单独编号，只传一次）；",
        "所以「开 PD」和「开 MTP」各自都要求对齐，两者叠加时对齐点更多。",
    ], "cloudd")

    # --- 本仓对照 ---------------------------------------------------------
    y4 = y3 + 192
    p.text(X, y4 - 32, 1300, 22,
           "<b>第三步：本仓三种 PD 包的差异（同一个模型，参数差很多）</b>", "text_b")
    from dg import table as t2
    rows2 = [
        ["sidecar / in_process", "TP=2，mml 65536，seqs 128", "TP=2 放得下模型 → 敢报长上下文"],
        ["dp2_tp1_194_195", "TP=1 + DP=2，mml 32768，seqs 32", "一卡一副本 → 上下文与并发都得压"],
        ["mooncake_pd2_tp1_195", "TP=1 + DP=2，<b>mml 512</b>，seqs 32，gpu 0.98",
         "为了「一卡塞下模型 + 开 MTP」把上下文压到极小"],
        ["mooncake_pd_195", "TP=2，seqs 128，D 侧手写 25 个 capture_sizes",
         "保留了参考实现的手工桶（都是 4 的倍数，对齐 K+1）"],
        ["fix_1120", "TP=8 + 专家并行 + 量化 + block 128，K=1",
         "DeepSeek 大模型必须切 8 卡；只抽 1 个 draft"],
    ]
    t2(p, X, y4, [("包", 340), ("关键参数", 620), ("为什么这么配", 460)], rows2,
       row_h=38)

    y5 = y4 + 38 * 6 + 36
    bridge(p, y5, "本页交接：PD 的硬约束讲完。下一步（⑦）把最容易混淆的一件事说清——"
                  "「enforce-eager」其实有两个作用域。")

    y6 = y5 + 76
    anchors(p, y6, [
        "块大小必须相等：vllm/v1/lwd_control/pd/cloud_runtime.py:219-221；"
        "vllm/v1/core/kv_cache_utils.py:717-741（hash_block_size = block_size × DCP）",
        "P/D TP 关系与切片：vllm-ascend/.../kv_pool/ascend_store/metadata.py:80-113；"
        "sfa_pd_rd2h/worker.py:743-756",
        "LWD 并行白名单与 DP 校验：vllm/distributed/lwd/lwd_parallel_groups.py:216-261；"
        "vllm/v1/engine/core.py:1336-1350",
        "P 侧请求改写：vllm/v1/lwd_control/pd/cloud_runtime.py:358-383",
        "长度预算：cloud_runtime.py:210-229",
        "拓扑必填项：vllm/config/lwd/topology.py:426-458",
        "本仓参数：各包 cloud_{p,d}.sh / decode.sh（见 ⑨ 页汇总）",
    ], h=128)

    y7 = y6 + 162
    concl(p, y7,
          "<b>一句话记住这一页：</b>PD 把「块大小、并行度、拓扑规模、端口」从建议变成硬约束，"
          "并强制改写 P 侧请求；<b>本仓三种 PD 包参数差得多，差别全在「一卡能不能放下整模型」。</b>")

    legend(p, y7 + 92)
