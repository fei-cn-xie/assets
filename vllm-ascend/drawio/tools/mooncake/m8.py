"""⑧ 什么时候算「成功」：三条状态线 + 失败兜底。

初学者最容易把「发出去了」当成「成功了」。这一页把三个层次的判定讲清楚：
块 → 请求 → 整条链路，以及每一层失败后会发生什么。
"""
from dg import Doc, legend_bar

TITLE = "⑧ 什么时候算「成功」：块 / 请求 / 链路 三层判定与失败兜底"
SUB = ("一句话：发出去了不算成功——要对端回执、要计数器归零、要块表里真的写进去了，"
       "才算这一单办完。")
X = 36


def build(doc: Doc) -> None:
    p = doc.page("⑧ 成功判定与兜底", pw=1560, ph=1320)
    p.title_block(TITLE, SUB, x=X, y=20, w=1500)

    # --- 三条状态线 -------------------------------------------------------
    y = 150
    p.text(X, y - 34, 1100, 22,
           "<b>第一层：块的状态（最底层，决定显存能不能回收）</b>", "text_b")
    b1 = p.box(X, y, 460, 96,
               "<b>空闲 / 已分配</b><br>块在自己的池子里，可用或正在被某个请求用着。",
               "box", "neutral")
    b2 = p.box(X + 500, y, 460, 96,
               "<b>被钉住（delay_free_blocks）</b><br>"
               "请求结束了但<b>还不能回收</b>：P 侧用 "
               "<font style=\"font-size:9px\">request_finished</font> "
               "返回 True 把它钉住，等 D 来拿。",
               "box", "cloudp")
    b3 = p.box(X + 1000, y, 460, 96,
               "<b>真正释放</b><br>"
               "D 回执到达、P 侧计数器归零后，"
               "<font style=\"font-size:9px\">get_finished</font> "
               "报给核芯，块才回到池子。",
               "box", "cloudd")
    p.edge(b1, b2, "请求结束", st="arrow", role="http")
    p.edge(b2, b3, "回执齐了", st="arrow", role="http")

    # --- 第二层：请求 -----------------------------------------------------
    y2 = y + 168
    p.text(X, y2 - 34, 1100, 22,
           "<b>第二层：请求的状态（调度器视角，决定「现在能不能算」）</b>",
           "text_b")
    r1 = p.box(X, y2, 700, 118,
               "<b>等 KV（WAITING）</b><br>"
               "D 侧已经知道「有 KV 在别人机器上」，"
               "调度器把它挂在等待队列，<b>先去跑别的请求</b>。"
               "<br><font style=\"font-size:9px\" color=\"#666666\">"
               "这就是 get_num_new_matched_tokens 返回第二个值 True 的含义</font>",
               "box", "neutral")
    r2 = p.box(X + 740, y2, 684, 118,
               "<b>可以算（RUNNING）</b><br>"
               "KV 到齐后，请求进入正常解码；"
               "此后每生成一个 token 都用到刚搬来的那段 KV。"
               "<br><font style=\"font-size:9px;color:#c0392b\">"
               "搬失败时<b>分两种口径</b>：原生路径把坏块记进 invalid_block_ids 让核心"
               "<b>重算</b>；而 <b>LWD PD 路径被强制成 fail-closed</b>"
               "（recompute_kv_load_failures=False）——直接终止请求，不重算、不换机器。</font>",
               "box", "cloudd")
    p.edge(r1, r2, "KV 到齐", st="arrow", role="http")

    # --- 第三层：链路 -----------------------------------------------------
    y3 = y2 + 190
    p.text(X, y3 - 34, 1200, 22,
           "<b>第三层：整条链路的「办完了」信号（谁说完成、谁相信）</b>",
           "text_b")
    l1 = p.box(X, y3, 690, 150,
               "<b>D 侧：我收完了（DONE_RECVING）</b><br>"
               "搬完之后 D 主动发一条完成消息给 P；"
               "<b>无论成功还是失败都要发</b>——<br>"
               "<font style=\"font-size:9px;color:#c0392b\">"
               "代码注释写明：不发会导致对端内存泄漏（P 的块永远钉着）。</font>",
               "box", "cloudd")
    l2 = p.box(X + 730, y3, 694, 150,
               "<b>P 侧：回执核销（ACK）</b><br>"
               "P 按「端口 + 片数」累计回执，"
               "达到 D 声明的片数才算这一单完成，然后回 ACK。<br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "注意：完成判定靠<b>计数器 + 轮询</b>，不是每个传输一个回调对象</font>",
               "box", "cloudp")
    p.edge(l1, l2, "", st="arrow", role="ctrl")

    # --- 失败兜底 ---------------------------------------------------------
    y4 = y3 + 212
    p.text(X, y4 - 34, 1100, 22,
           "<b>失败之后会怎样（四类兜底，按严重程度排）</b>", "text_b")
    f = [
        ("KV 导入失败（LWD PD）",
         "fail-closed：直接终止请求（日志 PD remote KV import failed），"
         "<b>不重算、不换 P/D</b>", "fail"),
        ("传输返回码 < 0", "抛 RuntimeError → 该请求标记失败 → 原生路径下这些块重算",
         "fail"),
        ("等不到 ACK", "打 error 日志后<b>吞掉异常</b>（不重试、不上抛）——"
                       "所以别指望它自愈", "fail"),
        ("对端根本不来拿", "480 秒后强制释放并打 error：Force freed expired request",
         "fail"),
        ("后台线程起不来", "启动时 5 分钟没 ready → RuntimeError 直接失败", "fail"),
    ]
    cy = y4
    for title, desc, role in f:
        p.box(X, cy, 1424, 58,
              f"<b>{title}</b>　→　<font style=\"font-size:10px\">{desc}</font>",
              "box", role)
        cy += 70

    # --- 结论 -------------------------------------------------------------
    y5 = cy + 24
    p.box(X, y5, 1424, 104,
          "<b>这一页的代码锚点</b><br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "钉住 / 延迟释放：kv_p2p/mooncake_connector.py:2058-2061（返回 delay_free_blocks）"
          "· :232-253（480 秒强制释放）<br>"
          "完成计数与轮询：:177（KVCacheTaskTracker）· :213-224 · :609"
          "（get_and_clear_finished_requests）<br>"
          "回执与 ACK：:373-397（P 侧核销）· :1502-1527（D 侧发 DONE_RECVING 并等 ACK）<br>"
          "失败 → 重算：:738-741、:628-631、:2789-2792（invalid_block_ids）<br>"
          "超时来源：vllm/vllm/envs.py:260（VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT，默认 480 s）"
          "</font>", "box", "neutral")

    y6 = y5 + 138
    p.box(X, y6, 1424, 60,
          "<b>一句话记住这一页：</b>「成功」= 数据落到对端显存 + 对端回了回执 + "
          "计数器归零 + 块真正释放。四件事差一件，都会变成显存泄漏或算错。",
          "box", "neutral")

    legend_bar(p, y6 + 94)
