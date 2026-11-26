"""② 一句话认识 Mooncake：三个角色、一个类比、两个部件。"""
from dg import Doc, legend_bar, table
TITLE = "② 一句话认识 Mooncake：把 KV 当成「可搬的货物」"
SUB = ("Mooncake 是月之暗面（Moonshot）开源的一套「以 KV 为中心」的推理基础设施。"
       "本页只讲清它由哪两块组成、我们把哪一块用在什么地方。")
X = 36
W = 470
GAP = 40
Y = 150


def cx(i: int) -> int:
    return X + i * (W + GAP)


def build(doc: Doc) -> None:
    p = doc.page("② Mooncake 是什么", pw=1560, ph=1180)
    p.title_block(TITLE, SUB, x=X, y=20, w=1500)

    # 三个角色
    p.text(X, Y - 34, 800, 22, "<b>第一步：认清这条链路上的三个角色</b>", "text_b")
    r1 = p.box(cx(0), Y, W, 120,
               "<b>P = Producer（生产者）</b><br>"
               "prefill 机器。它算出了 KV，手上有一批「货」要交给别人。<br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "代码里：prefill 实例 / 角色 role=cloud-P / 日志里的 P 侧</font>",
               "box", "cloudp")
    r2 = p.box(cx(1), Y, W, 120,
               "<b>D = Consumer（消费者）</b><br>"
               "decode 机器。它需要这批货才能继续生成下一个字。<br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "代码里：decode 实例 / 角色 role=cloud-D / 日志里的 D 侧</font>",
               "box", "cloudd")
    r3 = p.box(cx(2), Y, W, 120,
               "<b>搬运通道 = Transfer Engine</b><br>"
               "连接两台机器的「卡车」：只负责<b>把一段内存从 A 搬到 B</b>，"
               "不关心里面装的是什么。<br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "这才是本仓真正用到的那一块</font>",
               "box", "data")
    p.edge(r1, r2, "KV 从 P 交到 D", st="arrow", role="data")

    # 类比表
    y2 = Y + 190
    p.text(X, y2 - 34, 900, 22,
           "<b>第二步：用「寄快递」把这个过程整体类比一遍</b>", "text_b")
    rows = [
        ["寄件人打包 → 告诉快递公司包裹放哪", "把显存里那块 KV 的<b>地址和长度</b>注册给传输引擎",
         "register_memory"],
        ["快递公司知道收件人地址", "两台机器<b>交换</b>各自的内存段信息（谁在哪、多大）",
         "握手 / 建链"],
        ["下单发货", "发起一次<b>传输请求</b>：从我这的 X 地址，搬 N 字节到你的 Y 地址",
         "transfer / write / read"],
        ["收件人签收", "传完之后对方收到<b>回执</b>，才敢认为「KV 到了」",
         "完成通知 / 轮询"],
        ["包裹没到怎么办", "超时 → 放弃等待、报错（本项目<b>重算或直接失败</b>，不静默降级）",
         "timeout / fail"],
    ]
    table(p, X, y2, [("寄快递时……", 300), ("Mooncake 里对应发生什么", 620),
                     ("代码里的名字", 180)], rows, row_h=36)

    # 两个部件
    y3 = y2 + 250
    p.text(X, y3 - 34, 900, 22,
           "<b>第三步：Mooncake 其实有两个部件，别混为一谈</b>", "text_b")
    k1 = p.box(cx(0), y3, W, 170,
               "<b>① Transfer Engine（传输引擎）</b><br>"
               "纯搬运：注册内存 → 建连接 → 读/写一段字节。<br>"
               "它<b>不认识 KV</b>，你让它搬什么它就搬什么。<br><br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "类比：货运公司。你说是沙发还是服务器，它不管。</font>",
               "box", "data")
    k2 = p.box(cx(1), y3, W, 170,
               "<b>② Store / KV 池（以存换算）</b><br>"
               "把很多机器的显存+内存+SSD 组成一个大仓库，"
               "KV 可以<b>存进去、以后按 key 取出来</b>（跨请求复用）。<br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "类比：公共仓库。东西存在那，谁需要谁去取。</font>",
               "box", "neutral")
    k3 = p.box(cx(2), y3, W, 170,
               "<b>本仓用的是哪个？</b><br>"
               "只用 <b>① Transfer Engine</b>：P 与 D 直接点对点搬，"
               "KV 不进公共仓库。<br>"
               "<font style=\"font-size:9px\" color=\"#c0392b\">"
               "常见的误解：以为「KV 会先写进 Mooncake 池、再读出来」——本项目不是。</font>",
               "box", "fail")

    # 结论条
    y4 = y3 + 230
    p.box(X, y4, 1424, 58,
          "<b>一句话记住这一页：</b>Mooncake = "
          "「一个只会搬字节的传输引擎（本仓在用）」"
          "＋「一个可选的 KV 大仓库（本仓不用）」。"
          "P/D 分离里我们只用前者，把 KV 从 P 机显存直接搬到 D 机显存。",
          "box", "neutral")

    legend_bar(p, y4 + 92, extra_fill=("数据面 = 搬运通道", "data"))
