"""⑨ 五个常见误解 + 一本性能账（把「看起来懂」变成「真的懂」）。"""
from dg import Doc, legend_bar

TITLE = "⑨ 五个常见误解 ＋ 一本性能账"
SUB = ("这一页是前八页的「体检」：每条误解都给出正确的说法和代码位置。"
       "看完能自己回答「为什么这么设计」。")
X = 36


def build(doc: Doc) -> None:
    p = doc.page("⑨ 误解与性能账", pw=1560, ph=1420)
    p.title_block(TITLE, SUB, x=X, y=20, w=1500)

    # --- 五个误解 ---------------------------------------------------------
    y = 148
    p.text(X, y - 32, 900, 22, "<b>误解清单（★ = 最容易被读错的一个）</b>",
           "text_b")
    items = [
        ("★ 误解 1：P 算完会主动把 KV 推给 D",
         "实际：<b>D 主动去 P 读（pull）</b>。P 的「Sending」线程只回地址表、收完成信号，"
         "一字节 KV 都不推。",
         "mooncake_connector.py:333-417（只回元数据）vs :1042（D 端读）", "fail"),
        ("误解 2：Mooncake 是逐层搬运的",
         "实际：本项目<b>整请求块级批量</b>——一个请求的所有层、所有块，"
         "一次 batch_transfer_sync_read 搬完。三个 layerwise 钩子实现就是 pass。",
         ":1691-1703（三个 pass）；逐层版在 mooncake_layerwise_connector.py", "fail"),
        ("误解 3：KV 会先写进 Mooncake 的存储池",
         "实际：<b>点对点直接搬</b>，KV 不进任何公共池。"
         "Store 模式是另一套连接器（AscendStoreConnector），本 PD 场景不用。",
         "kv_transfer/__init__.py:46,57,74,80,87（五个连接器并列）", "fail"),
        ("误解 4：up_buffer_size / down_buffer_size 是 KV 容量",
         "实际：这是<b>另一个平面</b>的东西——host 数据面（UCX/共享内存）的帧块大小，"
         "跟 KV 传输无关。KV 走的是 kv_port。",
         "启动脚本 --kv-transfer-config 的 kv_port；数据面见 lwd_comm/data_comm/", "fail"),
        ("误解 5：显存地址随便给，反正引擎会自己找",
         "实际：① 必须<b>先注册再搬</b>，且每进程最多 256 段；"
         "② 某些路径（逐层版）要求<b>起始地址 2 MiB 对齐</b>，不对齐直接抛错。",
         "kv_transfer/utils/utils.py:15,455-472（段数上限）；:2556,2610（对齐断言）", "fail"),
    ]
    cy = y
    for title, body, anchor, role in items:
        h = 92
        p.box(X, cy, 1424, h,
              f"<b>{title}</b><br>"
              f"<font style=\"font-size:10px\">{body}</font><br>"
              f"<font style=\"font-size:9px\" color=\"#666666\">{anchor}</font>",
              "box", role)
        cy += h + 10

    # --- 性能账 -----------------------------------------------------------
    y2 = cy + 20
    p.text(X, y2 - 34, 1100, 22,
           "<b>一本性能账：决定「搬一次值不值」的几个量</b>", "text_b")
    cells = [
        ("一次搬的量", "该请求<b>所有层 × 所有块</b>的 KV<br>"
                       "（例：4K token、128 KB/token ≈ 512 MB）", "data"),
        ("搬运的并行度", "D 侧 32 线程池，按「对端 host:port」分桶；<br>"
                        "每个对端每轮最多 5 个请求就换人", "cloudd"),
        ("什么时候搬更划算", "prompt 长、输出短时最赚（K 值钱、搬一次用一次）；<br>"
                            "prompt 极短、输出极长时搬运占比反而高", "cloudp"),
        ("超时窗口", "对端 480 秒不来拿就强制释放；<br>"
                    "传输超时由 ASCEND_TRANSFER_TIMEOUT 决定", "fail"),
        ("并发窗口", "两侧用「有多少片要回执」来对账；<br>"
                    "P 的 TP 比 D 大时要分片拉、搬完再拼装", "neutral"),
    ]
    cw = 278
    for i, (title, body, role) in enumerate(cells):
        cid = p.box(X + i * (cw + 8), y2, cw, 168,
                    f"<b>{title}</b><br>"
                    f"<font style=\"font-size:10px\">{body}</font>", "box", role)
    p.text(X, y2 + 178, 1424, 40,
           "<font style=\"font-size:9px\" color=\"#666666\">"
           "数量级提示：cross-machine 带宽远低于显存带宽，所以「搬一次」的代价通常比"
           "「重算一遍 prefill」小得多——这正是 P/D 分离成立的经济学前提。</font>",
           "subtitle")

    # --- 结论 -------------------------------------------------------------
    y3 = y2 + 236
    p.box(X, y3, 1424, 88,
          "<b>一句话记住这一页：</b>这套实现的核心事实只有三条——"
          "①<b>D 拉、P 不推</b>；②<b>整请求块级批量，不逐层</b>；③<b>点对点，不进池</b>。"
          "其余细节（分片、拼装、对齐、超时）都是为这三条服务的。",
          "box", "neutral")

    legend_bar(p, y3 + 122)
