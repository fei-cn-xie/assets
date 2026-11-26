"""矩阵/张量绘图工具箱（给「Transformer 与模型文件结构」这套图用）。

设计约束（用户要求）：
  · 示例矩阵不能用 1~2 行/列，也不能用方阵（唯一例外：注意力分数天然是 token×token 方阵，
    届时要**明确标注为什么它必须是方阵**）；
  · 两个矩阵相乘时，左矩阵的**行数**要 ≠ 右矩阵的**列数**（这样读者能看出结果形状从哪来）；
  · 格子里放整数，并在页面上注明"真实模型里这些数是整数还是小数"。

配色语义（本套 21 页统一，与生命周期那套不同轴，图例里写明）：
  蓝 edge   = 数据 / 激活（token 向量、张量）
  橙 cloudp = 权重 / 参数（可训练矩阵）
  绿 cloudd = 结果 / 新产生的张量
  黄 data   = 计算动作（乘、softmax、采样）
  沙 neutral= 说明、结构、表格
  红 fail   = 易错点 / 常见误解
"""
from dg import C, Page, measure_h

ACT = "edge"        # 数据/激活
W = "cloudp"        # 权重/参数
OUT = "cloudd"      # 结果
CALC = "data"       # 计算动作
NOTE = "neutral"
ERR = "fail"

CW, CH = 40, 30     # 单个格子


# ------------------------------------------------------------------ 图例 ---
LEGEND = [
    ("数据 / 激活", ACT), ("权重 / 参数", W), ("结果 / 新张量", OUT),
    ("计算动作", CALC), ("说明 / 结构", NOTE), ("易错点", ERR),
]


def legend(page: Page, y: int) -> int:
    """本套图统一的图例（每页都要调，位置由调用方给）。"""
    page.text(36, y, 104, 20, "<b>图例</b>", "text_b")
    cx = 148
    for text, role in LEGEND:
        page.box(cx, y - 3, 120, 26, text, "box_center", role)
        cx += 128
    y2 = y + 34
    page.edge(src_pt=(148, y2 + 12), dst_pt=(204, y2 + 12), st="arrow_thick",
              role="http")
    page.text(210, y2, 220, 24, "数据流向 / 计算顺序", "subtitle")
    page.edge(src_pt=(448, y2 + 12), dst_pt=(504, y2 + 12),
              st="arrow_dash_open", role="http")
    page.text(510, y2, 300, 24, "虚线 = 形状/对照关系，不是数据", "subtitle")
    return y2 + 30


def title(page: Page, t: str, sub: str = "", w: int = 1600) -> None:
    page.title_block(t, sub, x=36, y=20, w=w)


# ---------------------------------------------------------------- 矩阵块 ---
def matrix(page: Page, x: int, y: int, data, role: str, name: str = "",
           dim: str = "", cw: int = CW, ch: int = CH,
           col_hdr=None, row_hdr=None, bold: set | None = None,
           fs: int = 11, hi: dict | None = None):
    """画一个矩阵。返回 (右边界x, 下边界y, 单元格 id 网格)。

    hi: {(i,j): role} —— 给个别格子单独上色（用来指出"这个数是这俩相乘加起来的"）。
    """
    bold = bold or set()
    hi = hi or {}
    ids = []
    m, n = len(data), len(data[0])
    if name:
        page.text(x, y - 46, max(180, n * cw), 20, f"<b>{name}</b>", "text_b")
    if dim:
        page.text(x, y - 26, max(220, n * cw), 20,
                  f"<font style=\"font-size:9px\" color=\"#666666\">{dim}</font>",
                  "subtitle")
    y0 = y + (22 if col_hdr else 0)
    for i, row in enumerate(data):
        for j, v in enumerate(row):
            r = hi.get((i, j), role)
            fill, stroke = C[r]
            cid = page.box(x + j * (cw + 2), y0 + i * (ch + 2), cw, ch,
                           f"<b>{v}</b>" if (i, j) in bold else str(v),
                           "box_center", None,
                           extra_style=(f"fillColor={fill};strokeColor={stroke};"
                                        f"fontSize={fs};"))
            ids.append((cid, i, j))
    if col_hdr:
        for j, h in enumerate(col_hdr):
            page.text(x + j * (cw + 2), y, cw, 20,
                      f"<font style=\"font-size:9px\" color=\"#666666\">{h}</font>",
                      "subtitle")
    if row_hdr:
        for i, h in enumerate(row_hdr):
            page.text(x - 76, y0 + i * (ch + 2), 70, ch,
                      f"<font style=\"font-size:9px\" color=\"#666666\">{h}</font>",
                      "subtitle")
    return (x + n * (cw + 2) - 2, y0 + m * (ch + 2) - 2, ids)


def op(page: Page, x: int, y: int, text: str, w: int = 44, h: int = 40,
       role: str = CALC) -> int:
    """矩阵之间的运算符 / 等号。返回操作符右边界。"""
    page.box(x, y, w, h, f"<b>{text}</b>", "box_center", role, extra_style="fontSize=16;")
    return x + w


def strip(page: Page, x: int, y: int, values, role: str, name: str = "",
          cw: int = 48, ch: int = 30, fs: int = 11):
    """一维序列（token id / 词片段）。**仅用于示意 shape，不做矩阵运算。**"""
    ids = []
    if name:
        page.text(x, y - 26, max(140, len(values) * (cw + 2)), 20,
                  f"<b>{name}</b>", "text_b")
    for i, v in enumerate(values):
        cid = page.box(x + i * (cw + 2), y, cw, ch,
                       f"<font style=\"font-size:{fs}px\">{v}</font>",
                       "box_center", role)
        ids.append(cid)
    return x + len(values) * (cw + 2) - 2, y + ch, ids


def panel(page: Page, x: int, y: int, w: int, title: str, body: str,
          role: str = NOTE, anchor: str = "") -> tuple[int, int]:
    """标题 + 正文，高度按字体度量自动算。

    注意：正文里常有显式 <br>，必须**逐段**估算行数；把整段当一个块会把高度算少
    （实测会溢出 2~4 行）。这个坑在 21 页里被复用，所以在这里一次修对。
    """
    from dg import est_lines

    inner = max(40, w - 30)
    n_title = est_lines(title, 11, inner)
    body_lines = 0
    for seg in body.split("<br>"):
        body_lines += est_lines(seg, 10, inner)
    n_anchor = 0
    if anchor:
        for seg in anchor.split("<br>"):
            n_anchor += est_lines(seg, 9, inner)
    h = 6 + 6 + n_title * 14.6 + body_lines * 13.4 + n_anchor * 12.0 + 6
    h = int(round(max(52, h)))

    label = f"<b>{title}</b><br><font style=\"font-size:10px\">{body}</font>"
    if anchor:
        label += (f"<br><font style=\"font-size:9px\" color=\"#666666\">"
                  f"{anchor}</font>")
    page.box(x, y, w, h, label, "box", role)
    return x + w, y + h


def callout(page: Page, x: int, y: int, w: int, text: str,
            role: str = NOTE, anchor: str = "") -> tuple[int, int]:
    """说明块：标题 + 正文 + 可选代码锚点。返回 (x2, y2)。"""
    label = text
    if anchor:
        label += (f"<br><font style=\"font-size:9px\" color=\"#666666\">"
                  f"{anchor}</font>")
    h = measure_h(text.split("<br>")[0], anchor, "<br>".join(text.split("<br>")[1:]),
                  w, min_h=44)
    page.box(x, y, w, h, label, "box", role)
    return x + w, y + h


def arrow(page: Page, x1, y1, x2, y2, label: str = "", role: str = "http",
          dashed: bool = False) -> None:
    page.edge(src_pt=(x1, y1), dst_pt=(x2, y2), label=label,
              st="arrow_dash" if dashed else "arrow", role=role)
