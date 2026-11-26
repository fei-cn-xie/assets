"""⑩ 术语表 + 三条阅读路径（从"看懂"到"能改代码"）。"""
from dg import Doc, legend_bar, table

TITLE = "⑩ 术语表与三条阅读路径"
SUB = ("这一页是工具页：卡在某个词就查表；想动手就按三条路径之一走。"
       "锚点按本仓当前检出分支（fix/lwd-multidp-dense-pd）。")
X = 36


def build(doc: Doc) -> None:
    p = doc.page("⑩ 术语表与阅读路径", pw=1560, ph=1480)
    p.title_block(TITLE, SUB, x=X, y=20, w=1500)

    # --- 术语表 -----------------------------------------------------------
    y = 146
    p.text(X, y - 32, 900, 22,
           "<b>术语表（左列是你会听到的词，右列是它的意思）</b>", "text_b")
    rows = [
        ["KV cache", "历史 token 的 K/V 缓存，省掉重复的矩阵乘法"],
        ["prefill / decode", "一次算完整段 prompt / 一个字一个字生成"],
        ["P / D", "prefill 机器（生产 KV）/ decode 机器（消费 KV）"],
        ["块（block）", "显存里固定大小的格子，放固定数量 token 的 KV"],
        ["块表（block table）", "「逻辑第几块 → 物理第几块」的映射"],
        ["块级 pull", "D 按块向 P 拉取 KV，一次拉一整个请求的所有层"],
        ["钉住（delay_free_blocks）", "P 请求结束后先不回收块，等 D 来拿（不是 pin API）"],
        ["回执（DONE_RECVING / ACK）", "D 说「我收完了」、P 说「收到，可以放块了」"],
        ["KV_READY（LWD 层）", "D 侧判定「KV 已到位」：收完块 + 计算起点对齐到 R"],
        ["Transfer Engine", "Mooncake 的搬运引擎，只认「已注册内存 + session + 地址长度」"],
        ["session", "对端标识，本项目里是「对端 IP : 对端引擎 rpc 端口」"],
        ["register_memory", "把一段内存登记给引擎（返回成功/失败，不返回地址）"],
        ["RDMA read", "网卡直接读对方显存，CPU 不参与搬数据"],
        ["P2PHANDSHAKE", "初始化时的 metadata_server 字面量：点对点自己握手，无需元数据服务"],
        ["fail-closed", "出问题就终止，不猜测、不降级（LWD PD 的口径）"],
        ["MTP / tail", "多 token 预测；tail 指 D 需要自己补算的那几个 token"],
    ]
    table(p, X, y, [("术语", 250), ("大白话解释", 620), ], rows, row_h=34)

    # --- 三条阅读路径 -----------------------------------------------------
    y2 = y + 34 * len(rows) + 60
    p.text(X, y2 - 32, 1100, 22,
           "<b>三条阅读路径：按你的目的选一条，别从第一行读到最后一行</b>",
           "text_b")
    path1 = p.box(X, y2, 466, 208,
                  "<b>路径 A：只想搞懂原理（不看代码）</b><br><br>"
                  "读 ① → ② → ③ → ⑦<br><br>"
                  "读完能讲清：<br>"
                  "· 为什么要有 KV cache<br>"
                  "· 为什么要跨机搬<br>"
                  "· 一次交接的四个阶段<br>"
                  "· 谁在等谁",
                  "box", "neutral")
    path2 = p.box(X + 500, y2, 460, 208,
                  "<b>路径 B：想改/调 connector</b><br><br>"
                  "读 ⑤ → ⑥ → ⑦ → ⑧<br><br>"
                  "重点看：<br>"
                  "· 钩子契约（base.py）<br>"
                  "· 调度侧 / worker 侧分工<br>"
                  "· 完成与失败判定<br>"
                  "· 改完先跑通「单请求」再谈性能",
                  "box", "cloudd")
    path3 = p.box(X + 994, y2, 430, 208,
                  "<b>路径 C：只想查/调性能</b><br><br>"
                  "读 ④ → ⑨ → ⑦<br><br>"
                  "关注：<br>"
                  "· 一次搬多少 MB<br>"
                  "· 对端分桶与线程池<br>"
                  "· 超时窗口（480 s / 传输超时）<br>"
                  "· 2 MiB 对齐与 256 段上限",
                  "box", "data")

    # --- 自己动手的 grep 清单 ----------------------------------------------
    y3 = y2 + 268
    p.text(X, y3 - 32, 1100, 22,
           "<b>自己动手：几条能直接抄的 grep（在 vllm-ascend 仓根目录跑）</b>",
           "text_b")
    p.box(X, y3, 1424, 200,
          "<font style=\"font-size:9px\" color=\"#333333\">"
          "<b># 谁在搬运（三套玩法都在这一层）</b><br>"
          "grep -rn \"batch_transfer_sync_read\" vllm_ascend/distributed/kv_transfer/<br>"
          "grep -rn \"batch_transfer_sync_write\" vllm_ascend/distributed/kv_transfer/<br><br>"
          "<b># 注册显存与引擎初始化</b><br>"
          "grep -rn \"register_memory\\|global_te\" "
          "vllm_ascend/distributed/kv_transfer/<br><br>"
          "<b># 端口是怎么算出来的（同一行里有 dp/tp/pp/pcp 四个因子）</b><br>"
          "grep -n \"handshake_port\\|side_channel_port\" "
          "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py<br><br>"
          "<b># LWD 层的 KV_READY 判定</b><br>"
          "grep -n \"finished_recving_kv_req_ids\\|tail_start\" "
          "vllm/v1/lwd_control/pd/cloud_scheduler.py"
          "</font>", "box", "neutral")

    # --- 一句话收尾 -------------------------------------------------------
    y4 = y3 + 200
    p.box(X, y4, 1424, 76,
          "<b>最后一句：</b>Mooncake 在本项目里干的是一件非常朴素的事——"
          "把 P 机显存里的几块字节，按 D 的请求搬到 D 的显存里，然后双方各自对账。"
          "所有复杂的东西（分片、拼装、对齐、超时、回执）都是为了让这件朴素的事"
          "<b>不出错、不漏显存、不阻塞调度</b>。",
          "box", "neutral")

    legend_bar(p, y4 + 110)
