"""过渡图那一套页面的共用小工具。"""
from dg import legend_bar

X = 36


def head(p, title: str, sub: str, w: int = 1500) -> None:
    p.title_block(title, sub, x=X, y=20, w=w)


def concl(p, y: int, text: str, w: int = 1424, role: str = "neutral") -> int:
    p.box(X, y, w, 58, text, "box", role)
    return y + 58


def anchors(p, y: int, lines: list[str], w: int = 1424, h: int = 96) -> int:
    body = "<br>".join(lines)
    p.box(X, y, w, h,
          "<b>代码锚点（想自己去翻的时候用）</b><br>"
          f"<font style=\"font-size:9px\" color=\"#666666\">{body}</font>",
          "box", "neutral")
    return y + h


def legend(p, y: int, extra_fill=None, extra_line=None) -> int:
    return legend_bar(p, y, extra_fill=extra_fill, extra_line=extra_line)


def panel(p, x, y, w, h, title, lines, role, font="font-size:10px"):
    p.box(x, y, w, h, "", "box", role)
    p.text(x + 10, y + 8, w - 20, 22, f"<b>{title}</b>", "text_b")
    body = "<br>".join(f"<font style=\"{font}\">{ln}</font>" for ln in lines)
    p.text(x + 10, y + 34, w - 20, h - 42, body, "text")


def bridge(p, y: int, text: str) -> int:
    """蓝底「本节交接」条：说明这一页接哪、下一站去哪。"""
    p.box(X, y, 1424, 46,
          f"<font style=\"font-size:10px\">{text}</font>", "box", "cloudd")
    return y + 46
