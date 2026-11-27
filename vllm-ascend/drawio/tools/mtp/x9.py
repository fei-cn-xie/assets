"""⑨ 本仓 LWD 下的 MTP：draft 在云侧跑，与 tail 交接如何共存。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc

TITLE = "⑨ LWD 边云 PD 下的 MTP：draft 在云侧，edge 只按接受数出 token"
SUB = ("一句话：<b>MTP 只在云端 P/D 两个实例上跑，edge 不跑</b>；"
       "而 PD 交接用 <font style=\"font-size:9px\">(p_end, i_end, r_pos)</font> 三元组定边界，"
       "D 从 <font style=\"font-size:9px\">r_pos</font> 起的「尾」才是它自己的。")


def build(doc: Doc) -> None:
    p = doc.page("⑨ LWD 下的 MTP", pw=1560, ph=1540)
    head(p, TITLE, SUB)
    X = 36

    # --- 谁跑 MTP ---------------------------------------------------------
    y = 152
    p.text(X, y - 34, 1300, 22,
           "<b>第一步：三个角色里只有两个跑 MTP</b>", "text_b")
    p.box(X, y, 460, 150,
          "<b>edge（1 卡）</b><br>"
          "<font style=\"font-size:10px\">"
          "没有 --speculative-config；模块归属表把 MTP 划给 cloud，edge 侧裁掉。<br>"
          "它只做：算 prompt embedding（UP 前）、按接受数做 UNEMBED（DOWN 后）。</font>",
          "box", "cloudp")
    p.box(X + 482, y, 460, 150,
          "<b>P（prefill）</b><br>"
          "<font style=\"font-size:10px\">"
          "配置里同样带 MTP，但 P 被压成 max_tokens=1、不产出用户可见输出；"
          "draft 在这里的实际收益有限（代码里没看到按角色跳过 draft 的守卫）。</font>",
          "box", "cloudd")
    p.box(X + 964, y, 460, 150,
          "<b>D（decode）</b><br>"
          "<font style=\"font-size:10px\">"
          "MTP 的主战场：每一步都起草 K 个、验证、把<b>真实接受行数</b>回传给 edge，"
          "由 edge 决定这次吐几个 token。</font>",
          "box", "cloudd")

    # --- 三元组 -----------------------------------------------------------
    y2 = y + 190
    p.text(X, y2 - 34, 1300, 22,
           "<b>第二步：PD 交接的三元组（MTP 让「边界」这件事更精确）</b>", "text_b")
    p.box(X, y2, 1424, 116,
          "<font style=\"font-size:10px\">"
          "<b>p_end</b> = P 真正算到的位置（只跑到 p_end+1）　·　"
          "<b>i_end</b> = KV 导入边界（D 只拉到这里）　·　"
          "<b>r_pos</b> = D 开始「拥有自己尾巴」的位置<br>"
          "代码里只允许两种合法形态（例如 truncate 与否），D 侧会<b>强校验</b>"
          "「P 导出的三元组」与本地算出的一致，不一致直接失败。<br>"
          "<font color=\"#c0392b\">有了 MTP，D 从 r_pos 起要「一次算 1+K 个位置」，"
          "所以 tail 的长度必须对得上 1+K 的节奏——这就是三元组必须精确的原因。</font></font>",
          "box", "neutral")

    # --- 接受数怎么回传 ---------------------------------------------------
    y3 = y2 + 156
    p.text(X, y3 - 34, 1300, 22,
           "<b>第三步：接受数如何穿过边云链路（最容易出错的一环）</b>", "text_b")
    panel(p, X, y3, 700, 176, "云侧：把「物理行」改写成「真实提交行数」", [
        "D 的一次前向输出是 1+K 行的数组（空位 -1）。",
        "回传给 edge 之前，云侧调度器把 num_accepted_tokens 从物理行数"
        "改写为<b>真实提交的行数</b>。",
        "否则 edge 会按 1+K 去取 hidden，取到 padding 的垃圾。",
    ], "fail")
    panel(p, X + 760, y3, 664, 176, "edge 侧：按接受数校验后 UNEMBED", [
        "edge 收到 DOWN 的 hidden 与接受数，强校验 end − start == num_accepted，",
        "再用 lm_head 做 UNEMBED，产出这次要吐给用户的 token。",
        "日志里能直接看到 accepted 数（[Lwd][edge-worker] UNEMBED … accepted=…）。",
    ], "cloudd")

    # --- draft 首步吃云侧 embedding ---------------------------------------
    y4 = y3 + 216
    p.text(X, y4 - 34, 1300, 22,
           "<b>第四步：LWD 特有的两处改动（只改 embedding 来源，不改起草逻辑）</b>",
           "text_b")
    p.box(X, y4, 1424, 104,
          "<font style=\"font-size:10px\">"
          "① <b>首步 embedding 来源</b>：正常路径靠 token id 查表；LWD 里云端已收到 edge 上传的 "
          "prompt embedding，所以 draft 首步直接复用那份张量（充填 <font style=\"font-size:9px\">"
          "inputs_embeds</font>）。<br>"
          "② <b>SP 下的切片</b>：外部 embedding 在序列并行下要按 TP 重新切一遍。<br>"
          "此外还有一条 guard：continued prefill 场景不允许回退成 token id 查表，"
          "缺权威 embedding 直接报错。</font>",
          "box", "cloudp")

    # --- 代价 -------------------------------------------------------------
    y5 = y4 + 144
    p.box(X, y5, 1424, 100,
          "<b>实测代价：开 MTP 会让并发变小</b><br>"
          "<font style=\"font-size:10px\">"
          "本仓 TP=1 且开 MTP 的部署包里，KV 只剩几十个块（32~63 个 GDN 块），"
          "max_num_seqs 只能设到 32；关掉 MTP 后可用 KV 明显回升。<br>"
          "原因就是 ⑦ 讲的两块开销：draft 自己的权重/缓冲 + lookahead 槽。"
          "<b>所以「开 MTP」不是无条件更快，要按接受率与并发一起权衡。</b></font>",
          "box", "fail")

    y6 = y5 + 140
    bridge(p, y6, "本页交接：整套讲完——概念（①②）、机制（③④⑤⑥）、约束（⑦）、落地（⑧⑨）。"
                  "下一步（⑩）给指标、日志与排障顺序。")

    y7 = y6 + 76
    anchors(p, y7, [
        "三元组计算与校验：vllm/v1/lwd_control/pd/envelope.py:215-227；"
        "vllm/v1/lwd_control/pd/cloud_runtime.py:226-227、:318-343、:356-357",
        "接受数改写：vllm/v1/lwd_control/control_cloud_scheduler/lwd_cloud_scheduler.py:534-546；"
        "edge 校验 vllm/v1/lwd_control/pd/edge_runtime.py:352-359",
        "LWD draft 改动：vllm-ascend/worker/lwd_cloud_model/lwd_mtp_proposer.py:28-59（首步 embedding）、"
        ":61-102（SP 切片）、:105（多 KV 组变体）",
        "谁来换 proposer：lwd_cloud_model_runner.py:284-300（条件：lwd_config 且 method=mtp）",
        "MTP 层 KV 单独编号：vllm-ascend/.../kv_p2p/mooncake_connector.py:2391-2403",
        "edge 不配 MTP：assets/c_home/bash/pds/30rc_1124/*/edge.sh（无 --speculative-config）",
    ], h=120)

    y8 = y7 + 154
    concl(p, y8,
          "<b>一句话记住这一页：</b>LWD 下 MTP 只在云侧跑；"
          "元组 (p_end, i_end, r_pos) 划清「P 算到哪、KV 拉到哪、D 从哪开始算」；"
          "接受数必须被改写成「真实提交行数」再回传，edge 才能正确 UNEMBED。")

    legend(p, y8 + 92, extra_fill=("数据面（DOWN/UNEMBED）", "data"))
