"""⑤ vLLM 给的「插头」：connector 必须实现哪些方法、什么时候被调用。

这一页只讲契约（不涉及 Mooncake 细节），是理解任意 KV connector 的地基。
"""
from dg import Doc, legend_bar

TITLE = "⑤ vLLM 给 connector 留的「插头」：钩子表与调用时机"
SUB = ("一句话：vLLM 不认识 Mooncake，它只在一个固定的时刻表里调用几个固定名字的方法。"
       "谁想搬 KV，就去实现这些方法。")
X = 36
W1 = 700


def build(doc: Doc) -> None:
    p = doc.page("⑤ connector 钩子表", pw=1560, ph=1300)
    p.title_block(TITLE, SUB, x=X, y=20, w=1500)

    # --- 左：调度侧 -------------------------------------------------------
    y = 150
    p.text(X, y - 34, 700, 22,
           "<b>调度侧（在 scheduler 进程里）：决定「要不要搬、搬哪些块」</b>",
           "text_b")
    left = [
        ("on_new_request（仅 P 侧）", "截：prompt 可能先丢掉最后一个 token",
         "让 D 自己重算最后一个 token，保证 tail 一定对得上"),
        ("get_num_new_matched_tokens", "问：这批 KV 能省掉多少重复计算？",
         "返回 (数量, 是否异步)。返回 >0 表示「有 KV 可以等」"),
        ("update_state_after_alloc", "记：我这些本地块要去远端拉",
         "把「本地块号 / 需要多少外部 token」登记下来"),
        ("build_connector_meta", "打包：把登记变成给 worker 的工单",
         "这一步之后才轮到 worker 侧真正搬"),
        ("request_finished（两侧都调）", "交：这一单怎么收尾",
         "P 返回 (True, 交接清单)＝块先别还；D 返回 (False, None)＝立刻还"),
    ]
    cy = y
    for name, what, extra in left:
        h = 84
        p.box(X, cy, W1, h,
              f"<b>{name}</b><br>"
              f"<font style=\"font-size:10px\">{what}</font><br>"
              f"<font style=\"font-size:9px\" color=\"#666666\">{extra}</font>",
              "box", "cloudp")
        cy += h + 12

    # --- 右：worker 侧 ----------------------------------------------------
    p.text(X + 780, y - 34, 640, 22,
           "<b>worker 侧（在跑模型的进程里）：真正动显存</b>", "text_b")
    right = [
        ("register_kv_caches", "注册：把我的 KV 显存登记给传输引擎",
         "顺便起后台线程（回元数据 / 收完成信号）", "cloudd"),
        ("bind_connector_metadata", "喂料：把调度侧的工单交给 worker 侧",
         "每次执行模型前调一次，用完即清", "cloudd"),
        ("start_load_kv", "下发：把工单交给后台搬运线程",
         "D 侧入口：切分片、挑端口、按对端分桶", "cloudd"),
        ("get_finished / get_transfer_results", "汇报：这一轮谁发完 / 谁收完",
         "核心拿到后才会真正释放块（解除「钉住」）", "cloudd"),
        ("get_block_ids_with_load_errors", "报错：哪些块没搬成功",
         "原生路径会对这些块<b>重算</b>；LWD PD 路径直接失败", "fail"),
    ]
    cy = y
    for name, what, extra, role in right:
        h = 84
        p.box(X + 780, cy, 644, h,
              f"<b>{name}</b><br>"
              f"<font style=\"font-size:10px\">{what}</font><br>"
              f"<font style=\"font-size:9px\" color=\"#666666\">{extra}</font>",
              "box", role)
        cy += h + 12

    # --- 中间：三个"躺平"的钩子 --------------------------------------------
    y2 = y + 5 * 96 + 30
    p.box(X, y2, 1424, 104,
          "<b>还有三个钩子，本项目里是「躺平」的（实现就是 pass）</b><br>"
          "wait_for_layer_load　/　save_kv_layer　/　wait_for_save<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "它们的存在是为了「逐层搬运」这种玩法：每算完一层就搬一层，早搬早省显存。"
          "本项目没用这个玩法——整请求算完再一次性块级 pull。"
          "逐层版在另一个文件（mooncake_layerwise_connector.py），别看混。</font>",
          "box", "neutral")

    # --- 时间线 -----------------------------------------------------------
    y3 = y2 + 140
    p.text(X, y3 - 34, 1200, 22,
           "<b>把钩子按时间顺序串起来（一次请求的一生）</b>", "text_b")
    steps = [
        ("请求进来", "neutral"), ("get_num_new_matched_tokens", "cloudd"),
        ("update_state_after_alloc", "cloudd"), ("build_connector_meta", "cloudd"),
        ("start_load_kv（后台线程搬）", "cloudd"), ("前向计算 / 生成 token", "neutral"),
        ("get_finished → 释放块", "cloudd"),
        ("request_finished（P 侧交接）", "cloudp"),
    ]
    cx = X
    for i, (txt, role) in enumerate(steps):
        w = 160
        p.box(cx, y3, w, 66, txt, "box_center", role)
        if i:
            p.edge(src_pt=(cx - 12, y3 + 33), dst_pt=(cx - 2, y3 + 33),
                   st="arrow", role="http")
        cx += w + 14

    y4 = y3 + 106
    p.box(X, y4, 1424, 118,
          "<b>为什么要有这么一套钩子（设计意图）</b><br>"
          "① <b>解耦</b>：vLLM 核心不需要知道 KV 是从网络来的、还是从磁盘来的；"
          "换 connector 就像换插头。<br>"
          "② <b>不阻塞调度</b>：「异步」这个返回值的意思是——我的 KV 还在路上，"
          "调度器可以先去算别的请求，等到了再用。<br>"
          "③ <b>可回退</b>：搬失败必须能报出来（get_block_ids_with_load_errors），"
          "让核心退化成「重新计算」，而不是用错的 KV 算出乱码。",
          "box", "neutral")

    y5 = y4 + 152
    p.box(X, y5, 1424, 78,
          "<b>这一页的代码锚点</b><br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "契约本体：vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py"
          "（KVConnectorBase_V1 的每个钩子都有 docstring）<br>"
          "本仓实现：vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py"
          "（调度侧 :1916 / :1958 / :1987 / :2016；worker 侧 :2627 / :3835 / :2767-2787）<br>"
          "注册表（名字 → 模块:类）：vllm-ascend/.../kv_transfer/__init__.py:46,57,74,80,87"
          "</font>", "box", "neutral")

    legend_bar(p, y5 + 108)
