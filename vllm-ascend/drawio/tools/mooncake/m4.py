"""④ 传输引擎怎么搬字节：注册 → 认路 → 下单（三步曲）。

再把三种玩法并列摆出来（块级 pull / 逐层 write / Store 池），
让读者明白"我们这套是哪一种、为什么"。
"""
from dg import Doc, legend_bar, table

TITLE = "④ 传输引擎怎么搬字节：注册 → 认路 → 下单（三步曲）"
SUB = ("一句话：传输引擎只认三件事——这块内存我登记过、对方是谁、从哪搬到哪搬多少。"
       "它不知道 KV 是什么，也不关心里面装的是什么。")
X = 36
W = 470
GAP = 42


def cx(i: int) -> int:
    return X + i * (W + GAP)


def build(doc: Doc) -> None:
    p = doc.page("④ 传输引擎三步曲", pw=1560, ph=1320)
    p.title_block(TITLE, SUB, x=X, y=20, w=1500)

    y = 150
    p.text(X, y - 34, 900, 22,
           "<b>三步曲：任何一次搬运都长这样</b>", "text_b")

    s1 = p.box(cx(0), y, W, 210, "<b>① 注册内存</b>", "box", "data")
    p.text(cx(0) + 10, y + 36, W - 20, 160,
           "<font style=\"font-size:10px\">"
           "把你的显存<b>登记</b>给引擎：<br>"
           "<font style=\"font-size:9px\" color=\"#666666\">"
           "register_memory(ptr, size)</font><br><br>"
           "登记之前，引擎不许碰这块内存（驱动要求）。<br><br>"
           "<font style=\"color:#c0392b\">注意：返回值是「成功/失败」，"
           "<b>不是地址</b>。地址要自己算：</font><br>"
           "<font style=\"font-size:9px\" color=\"#666666\">"
           "地址 = 层基址 + 块号 × 块步长</font><br><br>"
           "一笔显存会被切很多段登记，本项目上限 256 段"
           "（相邻 4096 B 以内的会先合并）"
           "</font>", "text")

    s2 = p.box(cx(1), y, W, 210, "<b>② 认路（认对方是谁）</b>", "box", "data")
    p.text(cx(1) + 10, y + 36, W - 20, 160,
           "<font style=\"font-size:10px\">"
           "传输要一个 <b>session</b> 标识，本项目里就是：<br>"
           "<font style=\"font-size:9px\" color=\"#666666\">"
           "session = \"对端IP : 对端引擎的 rpc 端口\"</font><br><br>"
           "启动时只做「握手」（换地址表），"
           "<b>真正的链路在第一次传输时才建立</b>。<br><br>"
           "<font style=\"color:#c0392b\">初学常见困惑：这里没有独立的 metadata server。</font><br>"
           "初始化时传的字面量 <font style=\"font-size:9px\">metadata_server=\"P2PHANDSHAKE\"</font> "
           "表示「点对点自己握手」。"
           "</font>", "text")

    s3 = p.box(cx(2), y, W, 210, "<b>③ 下单搬运</b>", "box", "data")
    p.text(cx(2) + 10, y + 36, W - 20, 160,
           "<font style=\"font-size:10px\">"
           "一次调用给三张<b>平行列表</b>：<br>"
           "<font style=\"font-size:9px\" color=\"#666666\">"
           "batch_transfer_sync_read(session,<br>"
           "&nbsp;&nbsp;src=[…], dst=[…], length=[…])</font><br><br>"
           "读 = 把对方的字节<b>拉到我这儿</b>；<br>"
           "写 = 把我这儿的字节<b>推到对方</b>。<br><br>"
           "名字里的 sync 指「函数返回时这批已经搬完」，"
           "<b>但它跑在后台线程里</b>，不会卡住推理主循环。"
           "</font>", "text")

    p.edge(s1, s2, "登记过才能搬", st="arrow", role="http")
    p.edge(s2, s3, "知道对方在哪", st="arrow", role="http")

    # --- 三种玩法对比 -----------------------------------------------------
    y2 = y + 268
    p.text(X, y2 - 34, 1100, 22,
           "<b>同一个引擎，三种玩法——本仓三套代码都在，别混着读</b>", "text_b")
    rows = [
        [("① 块级 pull（本项目默认）", "cloudd"),
         "D 主动去 P <b>读</b>整个请求的所有层",
         "一次 batch_transfer_sync_read；P 侧线程只回元数据"],
        [("② 逐层 push（layerwise 版）", "cloudp"),
         "P 每算完一层就<b>写</b>给对方",
         "batch_transfer_sync_write；TP 不对等时先用 128 B 预热建链"],
        [("③ Store 池（另一种连接器）", "neutral"),
         "KV 先变成「对象」存进大仓库，谁要谁去取",
         "需要 metadata/master 服务 + 内存池；本 PD 场景不用"],
    ]
    table(p, X, y2, [("玩法", 240), ("数据怎么流", 600), ("关键 API / 特征", 560)],
          rows, row_h=52)

    # --- 初学者容易卡住的三点 ----------------------------------------------
    y3 = y2 + 210
    p.box(X, y3, 1424, 118,
          "<b>三个最容易卡住的地方（都在代码里能看到）</b><br>"
          "① <b>协议不是 rdma / tcp</b>：昇腾版传的是 "
          "<font style=\"font-size:9px\">initialize(ip, \"P2PHANDSHAKE\", \"ascend\", device)</font>"
          "——第三个参数是 <b>ascend</b>。<br>"
          "② <b>地址不是「引擎给你的」</b>：登记只标记「这段可以搬」，"
          "搬运时用的是你自己算出来的 <font style=\"font-size:9px\">"
          "base + block_id × stride</font>。<br>"
          "③ <b>完成不是回调</b>：没有 per-transfer 的回调对象，"
          "靠<b>计数器 + 轮询</b>（搬完的请求进队列，主循环定期取）。",
          "box", "neutral")

    y4 = y3 + 152
    p.box(X, y4, 1424, 84,
          "<b>这一页的代码锚点</b><br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "单例初始化：vllm-ascend/.../kv_transfer/utils/mooncake_transfer_engine.py:11-43"
          "（initialize / register_memory / global_te）<br>"
          "注册 KV 显存：.../kv_p2p/mooncake_connector.py:2687-2690"
          "（先 395-452 合并区间，再逐段 register_memory）<br>"
          "真正搬运：同文件 :1042（batch_transfer_sync_read）· :825（session_id 拼法）<br>"
          "逐层版对照：.../kv_p2p/mooncake_layerwise_connector.py:557、:2115-2120"
          "</font>", "box", "neutral")

    y5 = y4 + 114
    p.box(X, y5, 1424, 60,
          "<b>一句话记住这一页：</b>传输引擎眼里没有 KV，只有"
          "「登记过的内存段 + 对端 session + 从哪到哪多长」；"
          "三步走完，一次搬运就结束了。",
          "box", "neutral")

    legend_bar(p, y5 + 94, extra_fill=("数据面 = 传输引擎", "data"))
