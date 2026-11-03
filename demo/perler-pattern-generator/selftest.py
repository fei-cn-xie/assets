# -*- coding: utf-8 -*-
"""selftest.py — 端到端自检：生成测试图 -> 转换 -> 校验一致性 -> 输出示例。"""

import os
import sys
import tempfile

from PIL import Image, ImageDraw

import core

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "selftest_output")


def make_test_image() -> Image.Image:
    img = Image.new("RGB", (300, 240), (250, 250, 250))
    d = ImageDraw.Draw(img)
    # 渐变背景
    for x in range(300):
        for y in range(0, 240, 2):
            d.point((x, y), fill=(int(x / 300 * 255), 80, int(200 - y / 240 * 200)))
    # 几何主体
    d.ellipse([60, 50, 180, 170], fill=(230, 70, 90), outline=(40, 40, 40), width=6)
    d.polygon([(150, 120), (250, 60), (250, 190)], fill=(255, 190, 20))
    d.rectangle([40, 180, 260, 225], fill=(30, 120, 200))
    return img


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    palette = core.load_palette()
    print(f"[1] 色板加载 OK：{len(palette)} 色")

    img = make_test_image()
    img.save(os.path.join(OUT, "input.png"))

    for name, kw in [("plain", dict(dither=False)), ("dither", dict(dither=True))]:
        pattern = core.convert(img, 58, 46, palette, **kw)
        # 一致性校验：网格尺寸、代码合法、豆数总和
        assert len(pattern.grid) == 46 and len(pattern.grid[0]) == 58, "网格尺寸错误"
        codes = {b.code for b in palette}
        assert all(c in codes for row in pattern.grid for c in row), "出现未知颜色代码"
        total = sum(n for _, n in pattern.counts())
        assert total == 58 * 46, f"豆数 {total} != {58*46}"
        sheet = core.render(pattern, cell=18, labels=True)
        flat = core.render_flat(pattern, cell=10)
        sheet.save(os.path.join(OUT, f"pattern_{name}.png"))
        flat.save(os.path.join(OUT, f"flat_{name}.png"))
        with open(os.path.join(OUT, f"counts_{name}.csv"), "w", encoding="utf-8-sig") as f:
            f.write(core.counts_csv(pattern))
        print(f"[2] {name:6s} 转换 OK：{pattern.total_beads()} 豆 / {len(pattern.counts())} 色 / "
              f"{pattern.boards()} 块小板")

    # 建议网格
    gw, gh = core.suggest_grid(img, 900)
    print(f"[3] 尺寸建议 OK：{gw}x{gh}")

    # Lab 往返健全性
    lab = core.rgb_to_lab(255, 0, 0)
    assert 50 < lab[0] < 60 and lab[1] > 60 and 50 < lab[2] < 75, f"红色 Lab 异常: {lab}"
    print(f"[4] Lab 校验 OK：red -> L*={lab[0]:.1f} a*={lab[1]:.1f} b*={lab[2]:.1f}")

    print("\n全部通过 ✔  示例输出见 selftest_output/ 目录")
    return 0


if __name__ == "__main__":
    sys.exit(main())
