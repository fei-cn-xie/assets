"""⑨ 本仓实测：只有一套包用 MoE，而 EP 与 TP 互斥。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc, table

TITLE = "⑨ 回到部署：本仓只有一套包用 MoE，也只有它开了 EP"
SUB = ("一句话：十套部署包里九套是 dense 模型（Qwen3.6-27B），<b>只有 DeepSeek 那套是 MoE 并开了 EP</b>；"
       "而 EP 与 TP 在 MoE 层里是<b>互斥</b>的——开 EP 就等于把 TP 关掉。")


def build(doc: Doc) -> None:
    p = doc.page("⑨ 部署与拓扑", pw=1560, ph=1560)
    head(p, TITLE, SUB)
    X = 36

    # --- 关键关系 ---------------------------------------------------------
    y = 146
    p.text(X, y - 30, 1300, 22,
           "<b>第一步：先记住这条互斥关系（它解释了本仓为什么这么配）</b>",
           "text_b")
    p.box(X, y, 1424, 132,
          "<font style=\"font-size:10px\">"
          "<b>开 EP 后，MoE 层把「专家」按卡切分 → 每张卡拥有<b>完整的一份专家</b> → "
          "这一层不再需要 TP 切分</b>。<br>"
          "代码里的表现：EP 打开时，MoE 层自己的 TP 规模被写成 <b>1</b>、EP 规模 = "
          "<b>DP × PCP × TP</b> 的乘积；EP 关闭时反过来（EP=1、TP=乘积）。<br>"
          "<font color=\"#c0392b\">所以「EP + TP 混合」是不支持的：你只能选「每卡几个专家」或"
          "「每卡整层切一半」，不能又切层又切专家。</font></font>",
          "box", "neutral")

    # --- 本仓实测表 -------------------------------------------------------
    y2 = y + 172
    p.text(X, y2 - 30, 1300, 22,
           "<b>第二步：十套包的实际配置（脚本实测）</b>", "text_b")
    rows = [
        ["Qwen3.6-27B（九套包）", "dense，没有专家可切", "无", "TP 1~2"],
        ["DeepSeek-V4-Flash-w8a8-mtp（fix_1120）", "<b>MoE + 量化 + MTP</b>",
         "<b>有（P/D 两侧都开）</b>", "TP=8 / PP=1"],
    ]
    table(p, X, y2, [("模型", 420), ("结构", 420), ("--enable-expert-parallel", 320),
                     ("并行", 264)], rows, row_h=44)
    p.box(X, y2 + 44 * 3 + 16, 1424, 108,
          "<font style=\"font-size:10px\">"
          "<b>读法</b>：dense 模型开 EP 是无效的（代码会先判「是不是 MoE」，不是就直接走无搬运路径），"
          "所以九套包不开也无所谓。<br>"
          "而 DeepSeek 那套 <b>单机 8 卡 TP=8 再开 EP</b>，实际含义是："
          "把 8 张卡当成<b>专家池</b>，每个 routed 专家只有一份、按卡分布 —— "
          "这样每卡不需要放「全部专家」，<b>单卡显存才放得下</b>。<br>"
          "<font color=\"#c0392b\">注意：脚本里没有写任何「专家数」参数，"
          "专家数与 top-k 全部来自模型目录的 config.json。</font></font>",
          "box", "cloudd")

    # --- 与 MTP 的关系 ----------------------------------------------------
    y3 = y2 + 44 * 3 + 164
    p.text(X, y3 - 30, 1300, 22,
           "<b>第三步：MoE 与 MTP / PD 的关系（三个容易搞错的点）</b>", "text_b")
    panel(p, X, y3, 460, 196, "① MTP 层本身就是 MoE 层", [
        "投机解码的草稿层不是「一个小 dense 层」，",
        "它内部<b>同样含一个完整的 MoE 层</b>（路由专家 + 共享专家）。",
        "所以开了 MTP 之后，MoE 的层数变多了——",
        "这也是「开 MTP 会让显存变紧」的原因之一。",
    ], "cloudd")
    panel(p, X + 482, y3, 460, 196, "② KV 传输与专家无关", [
        "KV cache 只挂在 attention 层，<b>MoE 层不产生 KV</b>。",
        "所以「专家怎么切」不影响 PD 分离要搬多少 KV。",
        "（那句「MTP 各层共用同一份 KV 层、只传一次」讲的是 MTP 内部复用，与 MoE 无关。）",
    ], "neutral")
    panel(p, X + 964, y3, 460, 196, "③ PD 与 EP 的真实耦合点是「专家 ID 表」", [
        "某些 MoE 模型（带 Hash 路由的那类）需要一张「token → 专家」的查表，"
        "由边侧算好、随上行载荷送到云端。",
        "而<b>通信方式一变（all-gather ↔ 按需分发），这张表的对齐方式也要跟着变</b>；"
        "对不齐会直接报错。这是 PD + EP 组合时最需要验证的一处。",
    ], "data")

    # --- 为什么本仓不开 EPLB -----------------------------------------------
    y4 = y3 + 236
    p.box(X, y4, 1424, 104,
          "<b>第四步：本仓为什么没有开 EPLB（如实说明）</b><br>"
          "<font style=\"font-size:10px\">"
          "十套包的启动参数里<b>从未出现</b> EPLB 相关字段（冗余专家数、负载均衡配置等）。"
          "可能原因：单机 TP=8 的专家池很小（每卡专家数不多），负载倾斜影响有限；"
          "且 EPLB 会额外占显存（冗余专家）——那套包本来显存就紧（还开着 MTP 与量化）。<br>"
          "<font color=\"#c0392b\">这属于「能开但权衡后没开」，不是「不能开」。</font></font>",
          "box", "fail")

    y5 = y4 + 144
    bridge(p, y5, "本页交接：最后一步（⑩）把术语、wave 协调（为什么 MoE 才需要）与排障清单收在一起。")

    y6 = y5 + 76
    anchors(p, y6, [
        "EP 与 TP 互斥（EP 开 → MoE 层 tp_size=1、ep_size=dp*pcp*tp）："
        "vllm/model_executor/layers/fused_moe/config.py:1212-1215、:1244-1259（原文注释 :1242-1243）",
        "专家按卡切分（linear / round_robin，非本卡 -1）：expert_map_manager.py:22-113",
        "EP 要求模型是 MoE：vllm/config/model.py:1373-1378（verify_with_parallel_config:1433 调用）",
        "本仓开 EP 的两处：assets/c_home/bash/pds/fix_1120/prefill.sh:112、decode.sh:112；"
        "模型路径 :13（DeepSeek-V4-Flash-w8a8-mtp）；TP=8 :102",
        "MTP 层含 MoE 层：vllm/model_executor/models/deepseek_mtp.py:235、:249-263"
        "（set_moe_parameters 逐个收集 MoE 层）",
        "KV 只在 attention 层、与专家无关：MoE 层不参与 KV 分配（见 生命周期套 ⑤）",
        "MTP 各层共用同一份 KV 层（与 MoE 无关）："
        "vllm-ascend/.../kv_p2p/mooncake_connector.py:554-558",
        "LWD 的专家 ID 表：边侧 lwd_edge_model_runner.py:284-294 → 云端 "
        "lwd_cloud_model_runner.py:814、:1250-1267 → 路由消费 "
        "router/fused_topk_router.py:165-216（按通信类型 all_gather / split）",
    ], h=160)

    y7 = y6 + 194
    concl(p, y7,
          "<b>一句话记住这一页：</b>本仓只有 MoE 那套包开 EP，而 EP 一旦打开就把 MoE 层的 TP 关掉；"
          "MTP 层也是 MoE 层、KV 与专家无关、PD 与 EP 的真实耦合点是那张「token→专家」表。"
          "<b>EPLB 是「能开但本仓权衡后没开」。</b>")

    legend(p, y7 + 92, extra_fill=("专家 ID 表/数据面", "data"))
