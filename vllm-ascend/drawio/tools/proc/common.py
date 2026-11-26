"""共用的小工具：统一页头、结论条、锚点块，免得每页重复写坐标。

所有页遵循同一编排：一句话结论（页头）→ 内容 → 结论条 → 代码锚点 → 图例。
"""
from dg import legend_bar

X = 36
SUB_Y = 48          # 副标题相对页头的偏移


def head(p, title: str, sub: str, w: int = 1500) -> None:
    p.title_block(title, sub, x=X, y=20, w=w)


def concl(p, y: int, text: str, w: int = 1424, role: str = "neutral") -> int:
    """结论条，返回它下方的 y。"""
    p.box(X, y, w, 58, text, "box", role)
    return y + 58


def anchors(p, y: int, lines: list[str], w: int = 1424, h: int = 96) -> int:
    """代码锚点块（9px 灰字），返回下方 y。"""
    body = "<br>".join(lines)
    p.box(X, y, w, h,
          "<b>代码锚点（想自己去翻的时候用）</b><br>"
          f"<font style=\"font-size:9px\" color=\"#666666\">{body}</font>",
          "box", "neutral")
    return y + h


def legend(p, y: int, extra_fill=None, extra_line=None) -> int:
    return legend_bar(p, y, extra_fill=extra_fill, extra_line=extra_line)
