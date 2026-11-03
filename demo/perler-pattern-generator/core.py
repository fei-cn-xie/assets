# -*- coding: utf-8 -*-
"""core.py — 拼豆图纸生成核心模块（无第三方依赖，仅用 Pillow）。

流程：读图 -> 缩放成 W x H 网格 -> (可选)增强 -> 映射到拼豆色板 -> 渲染图纸。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageEnhance

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PALETTE = os.path.join(HERE, "palette.json")


# ---------------------------------------------------------------- sRGB <-> Lab

def _srgb_to_linear(c: float) -> float:
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(v: float) -> float:
    v = max(0.0, min(1.0, v))
    s = v * 12.92 if v <= 0.0031308 else 1.055 * (v ** (1 / 2.4)) - 0.055
    return s * 255.0


def rgb_to_lab(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """sRGB (0-255) -> CIE L*a*b* (D65)."""
    rl, gl, bl = _srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b)
    x = (rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375) / 0.950456
    y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750
    z = (rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041) / 1.088754

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t) + 16 / 116

    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def _deltae76(lab1, lab2) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))


# ---------------------------------------------------------------- palette

@dataclass
class BeadColor:
    code: str
    name: str
    brand_code: str
    rgb: Tuple[int, int, int]
    lab: Tuple[float, float, float] = field(init=False)

    def __post_init__(self):
        self.lab = rgb_to_lab(*self.rgb)


def load_palette(path: str = DEFAULT_PALETTE) -> List[BeadColor]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for c in data["colors"]:
        hexs = c["hex"].lstrip("#")
        rgb = tuple(int(hexs[i : i + 2], 16) for i in (0, 2, 4))
        out.append(BeadColor(c["code"], c["name"], c.get("brand_code", ""), rgb))
    return out


# ---------------------------------------------------------------- convert

BOARD_SIZE = 29  # 标准小拼豆板 29 x 29


@dataclass
class Pattern:
    width: int
    height: int
    grid: List[List[str]]                       # grid[y][x] = palette code
    palette: List[BeadColor]
    transparent: set = field(default_factory=set)  # 空 格坐标集合 {(x, y)}

    def counts(self) -> List[Tuple[BeadColor, int]]:
        stat: Dict[str, int] = {}
        for row in self.grid:
            for code in row:
                if code is None:
                    continue
                stat[code] = stat.get(code, 0) + 1
        by_code = {b.code: b for b in self.palette}
        items = [(by_code[c], n) for c, n in stat.items() if c in by_code]
        items.sort(key=lambda t: -t[1])
        return items

    def total_beads(self) -> int:
        return sum(n for _, n in self.counts())

    def boards(self, board_size: int = BOARD_SIZE) -> int:
        bw = math.ceil(self.width / board_size)
        bh = math.ceil(self.height / board_size)
        return bw * bh


def _prepare(img: Image.Image, width: int, height: int,
             bg: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    img = img.convert("RGBA")
    bg = Image.new("RGBA", img.size, bg + (255,))
    img = Image.alpha_composite(bg, img).convert("RGB")
    return img.resize((width, height), Image.LANCZOS)


def convert(img: Image.Image, width: int, height: int,
            palette: List[BeadColor], dither: bool = False,
            brightness: float = 1.0, saturation: float = 1.0,
            contrast: float = 1.0) -> Pattern:
    """把任意图片转成 width x height 的拼豆网格。

    dither=True 时使用 Floyd-Steinberg 抖动，渐变更平滑但颜色更碎；
    False 时每格取最近色，色块干净、豆数统计更集中。
    """
    img = _prepare(img, width, height)
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if saturation != 1.0:
        img = ImageEnhance.Color(img).enhance(saturation)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)

    labs = [b.lab for b in palette]
    rgbs = [b.rgb for b in palette]
    px = img.load()

    grid: List[List[Optional[str]]] = [[None] * width for _ in range(height)]
    err = [[(0.0, 0.0, 0.0)] * width for _ in range(height)]

    def nearest(lab) -> int:
        best, bi = 1e18, 0
        for i, pl in enumerate(labs):
            d = _deltae76(lab, pl)
            if d < best:
                best, bi = d, i
        return bi

    for y in range(height):
        for x in range(width):
            r, g, b = px[x, y]
            lab = rgb_to_lab(r, g, b)
            if dither:
                ex, ey, ez = err[y][x]
                lab = (lab[0] + ex, lab[1] + ey, lab[2] + ez)
            i = nearest(lab)
            grid[y][x] = palette[i].code
            if dither:
                # 把 误差按 Floyd-Steinberg 分给邻居
                dr = lab[0] - labs[i][0]
                dg = lab[1] - labs[i][1]
                db = lab[2] - labs[i][2]
                for dx, dy, w in ((1, 0, 7 / 16), (-1, 1, 3 / 16), (0, 1, 5 / 16), (1, 1, 1 / 16)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        ex, ey, ez = err[ny][nx]
                        err[ny][nx] = (ex + dr * w, ey + dg * w, ez + db * w)

    return Pattern(width=width, height=height, grid=grid, palette=palette)


# ---------------------------------------------------------------- render

def _code_color(b: BeadColor) -> Tuple[int, int, int]:
    return b.rgb


def render(pattern: Pattern, cell: int = 20, grid_line: bool = True,
           labels: bool = True, symbol_mode: bool = False,
           label_every: int = 5) -> Image.Image:
    """渲染成图纸 PNG。

    cell        每格像素
    grid_line   画格线
    labels      标行列号（每 label_every 格标一次）
    symbol_mode 额外在格中心画符号，方便黑白打印对照
    """
    w, h = pattern.width, pattern.height
    margin = cell * 2 if labels else cell // 2
    iw = w * cell + margin * 2
    ih = h * cell + margin * 2
    img = Image.new("RGB", (iw, ih), "white")
    d = ImageDraw.Draw(img)
    by_code = {b.code: b for b in pattern.palette}

    for y in range(h):
        for x in range(w):
            code = pattern.grid[y][x]
            b = by_code.get(code)
            if b is None:
                continue
            x0, y0 = margin + x * cell, margin + y * cell
            d.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=_code_color(b))
            if symbol_mode:
                # 用亮度决定符号，黑白打印时也能区分深浅
                lum = 0.299 * b.rgb[0] + 0.587 * b.rgb[1] + 0.114 * b.rgb[2]
                d.line([x0 + cell * 0.25, y0 + cell * 0.5, x0 + cell * 0.75, y0 + cell * 0.5],
                       fill=(0, 0, 0) if lum < 128 else (255, 255, 255))

    if grid_line:
        for x in range(w + 1):
            px_ = margin + x * cell - (0 if x == 0 else 1)
            d.line([(px_, margin), (px_, ih - margin)], fill=(180, 180, 180))
        for y in range(h + 1):
            py = margin + y * cell - (0 if y == 0 else 1)
            d.line([(margin, py), (iw - margin, py)], fill=(180, 180, 180))

    if labels:
        fmin = max(9, cell // 2)
        try:
            from PIL import ImageFont
            font = ImageFont.load_default(size=fmin)
        except Exception:
            font = None
        for x in range(0, w, label_every):
            cx = margin + x * cell + cell // 2
            d.text((cx, 2), str(x + 1), fill=(60, 60, 60), anchor="mm", font=font)
            d.text((2, cx), str(x + 1), fill=(60, 60, 60), anchor="lm", font=font)
        for y in range(0, h, label_every):
            cy = margin + y * cell + cell // 2
            d.text((iw - 2, cy), str(y + 1), fill=(60, 60, 60), anchor="rm", font=font)
        # 每5格的主格线加粗，方便定位
        for x in range(0, w + 1, label_every):
            px_ = margin + x * cell - (0 if x == 0 else 1)
            d.line([(px_, margin), (px_, ih - margin)], fill=(120, 120, 120))
        for y in range(0, h + 1, label_every):
            py = margin + y * cell - (0 if y == 0 else 1)
            d.line([(margin, py), (iw - margin, py)], fill=(120, 120, 120))

    return img


def render_flat(pattern: Pattern, cell: int = 12) -> Image.Image:
    """无标注纯色块图，用于预览成品效果。"""
    w, h = pattern.width, pattern.height
    img = Image.new("RGB", (w * cell, h * cell), "white")
    d = ImageDraw.Draw(img)
    by_code = {b.code: b for b in pattern.palette}
    for y in range(h):
        for x in range(w):
            b = by_code.get(pattern.grid[y][x])
            if b is None:
                continue
            d.rectangle([x * cell, y * cell, x * cell + cell - 1, y * cell + cell - 1],
                        fill=b.rgb)
    return img


def counts_csv(pattern: Pattern) -> str:
    lines = ["编号,名称,厂号,数量"]
    total = 0
    for b, n in pattern.counts():
        lines.append(f"{b.code},{b.name},{b.brand_code},{n}")
        total += n
    lines.append(f"合计,,,  {total}")
    return "\n".join(lines)


def suggest_grid(img: Image.Image, target: int, board_size: int = BOARD_SIZE) -> Tuple[int, int]:
    """按原图长宽比给出贴近 target 网格数、且对齐拼板整数的建议尺寸。"""
    w, h = img.size
    ratio = h / w
    gw = max(board_size, round(math.sqrt(target / ratio)))
    gh = max(board_size, round(gw * ratio))
    # 对齐到板
    gw = math.ceil(gw / board_size) * board_size
    gh = math.ceil(gh / board_size) * board_size
    return gw, gh
