"""② 什么是「图」：捕获（录）与回放（播）。"""
from common import head, concl, anchors, legend, panel
from dg import Doc

TITLE = "② 什么是图：把「跑一遍」录下来，以后一次「播」"
SUB = ("一句话：<b>捕获</b>＝在专用 stream 上真正跑一遍 forward，"
       "同时把设备上发生的算子序列与地址关系录成图；"
       "<b>回放</b>＝一条命令重放整张图，CPU 不再逐个提交。")
X = 36


def build(doc: Doc) -> None:
    p = doc.page("② 捕获与回放", pw=1560, ph=1380)
    head(p, TITLE, SUB)

    # --- 三步对照 ---------------------------------------------------------
    y = 154
    p.text(X, y - 34, 1200, 22,
           "<b>第一步：三步看清它到底「录」了什么</b>", "text_b")

    panel(p, X, y, 466, 214, "① 普通执行：每次都重来一遍", [
        "Python 每步重新执行一遍模型代码：",
        "· 读当前的输入张量",
        "· 逐个调用算子（kernel/launch）",
        "· CPU 把每个算子提交给设备",
        "",
        "没有记忆：第 100 步和第 1 步做的事情完全一样，",
        "但 CPU 还是要把这上千次提交再做一遍。",
    ], "fail")

    panel(p, X + 490, y, 466, 214, "② 捕获：真跑一遍，同时录下来", [
        "在专用 stream 上放 <b>torch.cuda.graph(...)</b>（昇腾是 torch.npu.graph）：",
        "· 里面真的把 forward 跑了一次",
        "· 设备上发生的算子、依赖、地址，被录成一张图",
        "· 输入张量的<b>地址</b>也被记下（回放时必须一样）",
        "",
        "代价：这一步要真花钱（时间 + 一份内存池）。",
    ], "cloudp")

    panel(p, X + 980, y, 444, 214, "③ 回放：一句 replay()", [
        "以后每步只做：",
        "· 把新输入 <b>copy_</b> 到捕获时那几块固定 buffer 里",
        "· 调一次 <b>graph.replay()</b>",
        "· 从固定输出 buffer 里读结果",
        "",
        "CPU 只提交一次；设备按录好的顺序连着跑。",
        "",
        "注意：Python 代码<b>不会再执行</b>。",
    ], "cloudd")
    p.edge(src_pt=(X + 236, y + 234), dst_pt=(X + 723, y + 234), st="arrow",
           role="http")
    p.edge(src_pt=(X + 726, y + 234), dst_pt=(X + 1202, y + 234), st="arrow",
           role="http")

    # --- 一个关键性质 -----------------------------------------------------
    y2 = y + 274
    p.box(X, y2, 1424, 108,
          "<b>最关键的一条性质：录进去的只有「设备操作」，Python 逻辑不再执行</b><br>"
          "<font style=\"font-size:10px\">"
          "本仓文档原话：这段代码在捕获期间执行，"
          "<b>只有它发出的 GPU 操作会被记录并在回放时执行，它的 Python 本体不会再跑一遍</b>"
          "（vllm/docs/design/model_runner_v2.md:198）。<br>"
          "推论 1：图里的 <b>if / for</b> 只在捕获时决定一次，回放时不会重新判断——"
          "所以形状、分支、循环次数都必须固定。<br>"
          "推论 2：图里的 <b>.item() / print / 分配内存</b> 也只会发生一次，"
          "值会被「烤」进图里（后面 ④ 会讲这条坑）。</font>",
          "box", "neutral")

    # --- 类比 -------------------------------------------------------------
    y3 = y2 + 142
    p.text(X, y3 - 34, 1000, 22,
           "<b>第二步：一个类比（记住这个就够了）</b>", "text_b")
    panel(p, X, y3, 700, 150, "像「宏」（macro）/ 像「录屏回放」", [
        "· 普通执行 = 每一步都手工点一遍菜单；",
        "· 捕获 = 手工点一次，把动作序列录成一个宏；",
        "· 回放 = 以后按一下宏的快捷键，它把整串动作重放。",
        "宏能成立的前提：<b>每次点的位置必须一样</b>（＝地址固定）、"
        "<b>菜单结构不能变</b>（＝形状固定）。",
    ], "neutral")
    panel(p, X + 760, y3, 664, 150, "它「不是」什么（三个常见误解）", [
        "· 不是「把模型编译成更快的算子」——那是 torch.compile 干的事；",
        "· 不是「缓存计算结果」——它不省计算，只省「提交」；",
        "· 不是「自动加速一切」——形状一变就要换一张图，没有对应图就退回普通执行。",
    ], "fail")

    y4 = y3 + 190
    anchors(p, y4, [
        "捕获的真实代码（跑一遍并录）：vllm/compilation/cuda_graph.py:315-321"
        "（with torch.cuda.graph(...): output = self.runnable(...)）",
        "回放就一行：vllm/compilation/cuda_graph.py:362（entry.cudagraph.replay()）",
        "「只录 GPU 操作、Python 不再跑」：vllm/docs/design/model_runner_v2.md:198",
        "录下来的产物是「零参可调用列表」：vllm/compilation/breakable_cudagraph.py:3-21",
        "Ascend 上的对应实现：vllm-ascend/vllm_ascend/compilation/acl_graph.py"
        "（捕获与回放各一处）",
    ], h=104)

    y5 = y4 + 136
    concl(p, y5,
          "<b>一句话记住这一页：</b>捕获 = 「真跑一遍顺便录下来」，回放 = 「一条命令重放」；"
          "它省的是 CPU 的提交次数，不是计算量——"
          "所以前提是「形状固定、地址固定、图里不能有 CPU 同步」。")

    legend(p, y5 + 92)
