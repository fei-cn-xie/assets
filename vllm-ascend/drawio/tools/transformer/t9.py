"""⑨ 加权求和：O = P·V（5×5 乘 5×3 得 5×3）。"""
import ex
from mth import (ACT, CALC, ERR, NOTE, OUT, W, legend, matrix, op, panel, title)

TITLE = "⑨ 加权求和：O = P · V（每个 token 把别人的内容按比例搬过来）"
SUB = ("这是注意力块的最后一步，也是一次普通矩阵乘：概率矩阵 P（5×5）乘内容矩阵 V（5×3）。"
       "结果 O 就是「这一层注意力加工后的新表示」。")


def build(doc) -> None:
    p = doc.page("⑨ 加权求和", pw=1700, ph=1300)
    title(p, TITLE, SUB, w=1620)

    # --- 一句话 -----------------------------------------------------------
    y = 150
    panel(p, 36, y, 1628, "一句话：P 是「配料比例」，V 是「食材」，O 是「拌好的菜」",
          "P 的第 i 行说「第 i 个 token 该按什么比例参考别人」，"
          "V 的第 j 行是「第 j 个 token 的内容」。两者相乘 = 按比例把内容混起来。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "文字描述就是这样发生的：模型想让「它」指代前面的「猫」时，"
          "只要把「猫」那一列的权重调大，内容就自动被搬进来了。</font>",
          CALC)

    # --- 三个矩阵 ---------------------------------------------------------
    y2 = y + 108
    p.text(36, y2 - 60, 1200, 22,
           "<b>第一步：摆出来（P 来自 ⑧，V 来自 ⑥）</b>", "text_b")
    px, py, _ = matrix(p, 60, y2 + 6, ex.P, OUT, name="P（权重）",
                       dim="5×5", cw=46)
    ox = op(p, px + 14, y2 + 46, "·", 40, 40)
    vx, vy, _ = matrix(p, ox + 14, y2 + 6, ex.V, ACT, name="V（内容）",
                       dim="5×3")
    ox2 = op(p, vx + 14, y2 + 46, "=", 40, 40)
    ox3, oy3, _ = matrix(p, ox2 + 14, y2 + 6, ex.O, CALC, name="O（输出）",
                         dim="5×3　又是「行=token」，形状回到 5×3")

    # --- 手算一格 ---------------------------------------------------------
    y3 = y2 + 230
    p.text(36, y3 - 60, 1200, 22,
           "<b>第二步：手算 O[0][0]（第 0 个 token 的新特征第 0 维）</b>", "text_b")
    terms = " + ".join(f"{ex.P[0][t]}×{ex.V[t][0]}" for t in range(5))
    panel(p, 36, y3 + 6, 900, "把「第 0 行 P」和「第 0 列 V」逐项乘再相加",
          f"<font style=\"font-size:11px\">P 第 0 行 = {ex.P[0]}<br>"
          f"V 第 0 列 = {[ex.V[t][0] for t in range(5)]}</font><br>"
          f"<font style=\"font-size:10px\">{terms} = <b>{ex.O[0][0]}</b></font><br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "权重 0.62 的项贡献最大——注意力学到的东西，最后就落在这些系数上。</font>",
          CALC)
    panel(p, 966, y3 + 6, 698, "为什么这一步之后可以「忘掉」P",
          "O 已经把「注意谁」的信息压进了数值里，下游不需要再知道 P 长什么样。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "这也是为什么推理框架只缓存 K/V、不缓存 P："
          "P 每来一个新 token 都要重算，K/V 不用（这就是 KV cache 的由来）。</font>",
          NOTE)

    # --- 形状回读 ---------------------------------------------------------
    y4 = y3 + 190
    p.text(36, y4 - 60, 1200, 22,
           "<b>第三步：核对形状（这一步最容易算错）</b>", "text_b")
    rows = [
        ["Q", "5×3", "5 个 token，每个 3 维 query"],
        ["Kᵀ", "3×5", "K 转置；3 = 每头维度"],
        ["S = Q·Kᵀ", "5×5", "行=发问者，列=被关注者（方阵，不可避免）"],
        ["P = softmax(S/√d)", "5×5", "同样的形状，值变成 0~1"],
        ["V", "5×3", "5 个 token 各自的内容"],
        ["O = P·V", "5×3", "回到「行=token，列=特征」——和 X 一样"],
    ]
    from dg import table
    table(p, 36, y4 + 6, [("量", 240), ("形状", 120), ("含义", 620)],
          rows, row_h=34)

    p.box(1050, y4 + 6, 614, 6 * 34 + 34,
          "<b>一条捷径（老手都在用）</b><br>"
          "整个注意力块可以只记两句话：<br>"
          "① 形状上：<font style=\"font-size:9px\">"
          "(n×d) → S (n×n) → O (n×d)</font>，<b>进出不变</b>；<br>"
          "② 含义上：先算「谁看谁」，再把「内容」按权重搬过来。<br><br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "「n×d → n×n → n×d」这个往返，是后面残差连接能成立的前提（⑪）。</font>",
          "box", NOTE)

    # --- 代码锚点 --------------------------------------------------------
    y5 = y4 + 290
    p.box(36, y5, 1628, 96,
          "<b>代码锚点</b><br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "这一乘在 attention backend 里和 softmax 融合在一起（fused），"
          "所以你不会看到独立的「P·V」调用：vllm/v1/attention/backends/ 下各后端<br>"
          "输出投影（把各头的 O 拼起来再乘 W_O）："
          "vllm/model_executor/models/llama.py:175（self.o_proj = RowParallelLinear(...)）<br>"
          "KV cache 的由来：只缓存 K/V，不缓存 P —— "
          "vllm/v1/core/kv_cache_utils.py（KVCacheSpec）与 attention backend 的 KV 读写"
          "</font>", "box", NOTE)

    legend(p, y5 + 132)
