"""Audit the colours actually written into a .drawio against the palette rules.

每页做四件事：
  1. 统计填色分布（白/角色色/灰阶/数据面），并给出"是否几乎没有色"的判断；
  2. 报出**不在规范调色板里的历史遗留色**（改版后没清干净的）；
  3. 检查线条颜色是否只来自平面色；
  4. 检查每页都有且只有一个统一图例，且图例里的色块用的是规范色。

用法： python color_audit.py <file.drawio> [--verbose]
"""
from __future__ import annotations

import collections
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:                                   # 重定向时也别变成 GBK，否则表格全是乱码
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
except Exception:
    pass

import dg                                       # noqa: E402
from render_check import load, parse_style      # noqa: E402

# 填色 → 语义（只列本项目在用的；不存在"白色合法"这一项）
FILL_ROLE = {
    "#dae8fc": "边端 edge",
    "#ffe6cc": "云 P",
    "#d5e8d4": "云 D",
    "#b0e3e6": "PD Proxy",
    "#f8cecc": "失败/取消",
    "#fff2cc": "数据面(黄)",
    "#eceff1": "客户端",
    "#efe9dd": "中立/通用",
    "#e7e0d4": "阶段分组带",
    "#f4f4f4": "灰阶 proc1",
    "#e8e8e8": "灰阶 proc2",
    "#dcdcdc": "灰阶 proc3",
    "#d0d0d0": "灰阶 proc4",
    "#c4c4c4": "灰阶 proc5",
}
# 明确不允许的填色：白色（看色差时看不出来）、以及改版淘汰的旧色
FORBIDDEN_FILL = {
    "#ffffff": "白色（区分不出「本来就中立」还是「忘了上色」）",
    "#ffd9b3": "旧·云P深橙",
    "#e1d5e7": "旧·控制面紫(当填充用)",
    "#eeeeee": "旧·灰",
    "#f5f5f5": "旧·近白",
}
LINE_ROLE = {
    "#9673a6": "控制面 ZMQ", "#d6b656": "数据面", "#666666": "HTTP/普通",
    "#9e9e9e": "灰阶描边", "#8a8a8a": "灰阶描边", "#757575": "灰阶描边",
    "#b3b3b3": "中立描边", "#a1887f": "中立描边", "#b85450": "失败",
    "#6c8ebf": "边端", "#d79b00": "云 P", "#82b366": "云 D", "#0e8088": "PD Proxy",
    "#999999": "注记", "#607d8b": "客户端", "#000000": "默认黑",
}


def role_families(label: str) -> set[str]:
    """**标题行**里提到了哪些角色（用于"文案说边端、颜色却不是"的一致性检查）。

    只看标题行：后面的补充说明/锚点经常"顺带提到对端"（例如云 P 的框里写
    "视觉塔留在边侧"），那不是归属声明，按它判会大量误报。
    `prefill` / `decode` 要整词匹配：`do_remote_prefill=True` 这类字段名不是归属。
    """
    import re
    title = re.sub(r"<[^>]+>", "", label.split("<br>")[0])
    fam = set()
    if re.search(r"边端|边侧", title):
        fam.add("edge")
    if re.search(r"云 ?P(?![a-zA-Z])|(?<![_A-Za-z])prefill(?![_A-Za-z])|Producer",
                 title):
        fam.add("cloudp")
    if re.search(r"云 ?D(?![a-zA-Z])|(?<![_A-Za-z])decode(?![_A-Za-z])|Consumer",
                 title):
        fam.add("cloudd")
    if re.search(r"PD Proxy|proxy", title):
        fam.add("proxy")
    return fam


FAMILY_FILL = {"edge": "#dae8fc", "cloudp": "#ffe6cc", "cloudd": "#d5e8d4",
               "proxy": "#b0e3e6"}


def audit(path: str, verbose: bool = False) -> int:
    pages = load(path)
    problems: list[str] = []
    hints: list[str] = []
    scanned = 0
    print(f"{os.path.basename(path)}: {len(pages)} pages\n")
    print(f"{'页':<26} {'方块':>4} {'边':>4}  填色分布")
    print("-" * 96)

    for pi, page in enumerate(pages):
        fills = collections.Counter()
        bad_fill = []
        bad_line = []
        # 图例色块不参与"内容"统计（否则每页都会凭空多出 7 个角色色）
        legend_y = [v.y for v in page["verts"].values()
                    if v.label.replace("<b>", "").replace("</b>", "").strip() == "图例"]
        ly = legend_y[0] if legend_y else None
        lanes = [(v.id, v.x, v.y, v.w, (v.style.get("fillColor") or "").lower())
                 for v in page["verts"].values()
                 if v.w > 300 and v.h > 300 and "opacity=45" in v.style]
        same_as_lane = 0
        for v in page["verts"].values():
            if v.w <= 0 or not v.label:
                continue
            st = v.style
            fill = (st.get("fillColor") or "#ffffff").lower()
            if st.get("__shape__") == "text" or fill == "none":
                continue
            if ly is not None and ly - 6 <= v.y <= ly + 30 and 100 <= v.w <= 260:
                continue                      # 图例色块
            name = FILL_ROLE.get(fill)
            if fill in FORBIDDEN_FILL:
                bad_fill.append((v.id, fill, FORBIDDEN_FILL[fill], v.label[:44]))
                name = f"✗{fill}"
            elif name is None:
                bad_fill.append((v.id, fill, "未登记", v.label[:44]))
                name = f"?{fill}"
            fills[name] += 1
            # 文案↔填色一致性（启发式，只扫"内容方框"：圆角 + 带 .py: 锚点）
            # 内容方框 = 圆角 + 带 .py 锚点；排除表格 cell（font-size:11px 包裹）
            is_content = (v.style.get("rounded") == "1"
                          and ".py" in v.label
                          and "font-size:11px" not in v.label)
            fam = role_families(v.label)
            if is_content:
                scanned += 1
            if is_content and len(fam) == 1:
                want = FAMILY_FILL[next(iter(fam))]
                if fill != want and fill != "#f8cecc":
                    hints.append(
                        f"[{pi:02d} {page['name']}] 文案只提「{next(iter(fam))}」，"
                        f"填色却是 {FILL_ROLE.get(fill, fill)} :: "
                        f"{v.label.split('<br>')[0][:46]}")
            for _lid, lx, ly_, lw, lfill in lanes:
                if lx - 2 <= v.x <= lx + lw and ly_ - 2 <= v.y <= ly_ + 4000 \
                        and lfill == fill:
                    same_as_lane += 1
                    break
        for e in page["edges"].values():
            col = (e.style.get("strokeColor") or "#000000").lower()
            if col not in LINE_ROLE:
                bad_line.append((e.id, col))

        total = sum(fills.values())
        neutralish = sum(c for n, c in fills.items() if "中立" in n)
        greys = sum(c for n, c in fills.items() if "灰阶" in n)
        summary = "  ".join(f"{n}×{c}" for n, c in fills.most_common())
        print(f"[{pi:02d}] {page['name'][:22]:<22} {total:>4} {len(page['edges']):>4}  {summary}")

        # 只有"几乎全是中立色、且没有灰阶分层"才判为发白（P09 那种三列灰阶是合法的）
        if total and not greys and neutralish / total > 0.85 and total > 8:
            problems.append(f"[{pi:02d} {page['name']}] {neutralish}/{total} 个方块是中立色、"
                            f"又没有灰阶分层，看图会发白")
        if same_as_lane:
            problems.append(f"[{pi:02d} {page['name']}] {same_as_lane} 个方块与所在泳道同色，"
                            f"层次看不出来（泳道给底色、方块给中立色）")
        for cid, fill, stroke, label in bad_fill:
            problems.append(f"[{pi:02d}] 非规范填色 {cid} {fill} {stroke} :: {label}")
        for cid, col in bad_line:
            problems.append(f"[{pi:02d}] 非规范线色 {cid} {col}")

        # 图例检查：必须恰好一个 "图例" 文本，且图例区色块齐全
        legends = [v for v in page["verts"].values()
                   if v.label.replace("<b>", "").replace("</b>", "").strip() == "图例"]
        if len(legends) != 1:
            problems.append(f"[{pi:02d} {page['name']}] 图例数量={len(legends)}（应为 1）")
        else:
            ly = legends[0].y
            chips = [v for v in page["verts"].values()
                     if abs(v.y - (ly - 3)) < 6 and 100 <= v.w <= 260 and v.label]
            got = {(v.style.get("fillColor") or "").lower() for v in chips}
            want = {c[0].lower() for c in dg.LEGEND_ITEMS_ROLES}
            missing = want - got
            if missing:
                problems.append(f"[{pi:02d} {page['name']}] 图例缺色块: {sorted(missing)}")
            if verbose:
                print(f"      图例色块 {len(chips)} 个:", sorted(got))

    print()
    if hints:
        print(f"{len(hints)} 条「文案只提一个角色、填色却不是它」的候选"
              f"（启发式，逐条判断：有些步骤本就该中立）：")
        for h in hints:
            print("  ?", h)
        print()
    else:
        print(f"角色一致性：扫描了 {scanned} 个内容方框，"
              f"没有「文案说某角色、颜色却不是它」的情况")
    if problems:
        print(f"{len(problems)} 个待处理项：")
        for p in problems:
            print("  -", p)
        return 1
    print("OK: 所有页面的填色/线色/图例都符合调色板规则")
    return 0


if __name__ == "__main__":
    raise SystemExit(audit(sys.argv[1], "--verbose" in sys.argv))
