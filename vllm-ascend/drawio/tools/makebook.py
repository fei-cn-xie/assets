"""Build a browsable single-volume book out of all diagram sets.

    cd assets/vllm-ascend/drawio/tools
    python makebook.py                 # writes ../book/
    python makebook.py --scale 0.85

Output:
    book/index.html          master index (reading order + per-set entry)
    book/<set>.html          one page per set: every diagram as inline SVG,
                             with its 一句话结论 and 代码锚点 extracted from the diagram
    book/README.md           plain-text index (for grep / GitHub)

Why SVG: the same geometry the checker already parses, so the book can never
drift from the .drawio files. Regenerate after any diagram change.
"""
from __future__ import annotations

import html
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DRAWIO = os.path.dirname(HERE)
BOOK = os.path.join(DRAWIO, "book")

sys.path.insert(0, HERE)
import render_check as rc  # noqa: E402

# 阅读顺序（总纲里的十站）；文件 → (站号, 显示名, 一句话定位)
SETS = [
    ("Transformer与模型文件结构.drawio",
     "站1", "模型本身怎么算 · 21 页",
     "一次 forward 里每一步在算什么矩阵；权重文件长什么样"),
    ("vLLM训练侧与权重来源.drawio",
     "站0", "权重从哪来 · 10 页",
     "训练九步、显存四件套、量化、MTP 两义、加载器挑文件、RL 后端"),
    ("vLLM入门-从训练到推理系统.drawio",
     "站2", "系统为什么这么设计 · 10 页",
     "为什么 decode 卡带宽、为什么要 batching、显存账、四种并行"),
    ("vLLM生命周期-端到端.drawio",
     "站3", "一次部署的一生 · 9 页",
     "启动、模型加载、KV cache、单请求、PD 交接、Step 循环"),
    ("vLLM进程与线程.drawio",
     "站4", "运行时骨架 · 16 页",
     "几个进程/线程、ZMQ 三通道、调度器、dp2tp4 流转、权重参与计算"),
    ("Mooncake原理与实现.drawio",
     "站5", "KV 怎么跨机搬 · 10 页",
     "传输引擎三步曲、块级 pull、回执、失败兜底"),
    ("vLLM图捕获.drawio",
     "站6", "怎么把提交开销压掉 · 10 页",
     "捕获与回放、三个硬约束、四种模式、桶与 dispatch、昇腾 ACL Graph"),
    ("vLLM的MTP与投机解码.drawio",
     "站7", "怎么让一次前向多产出几个 token · 10 页",
     "draft 四种来源、拒绝采样、KV 回退、三条约束、LWD 适配"),
    ("vLLM的MoE与专家并行.drawio",
     "站10", "混合专家怎么算 · 10 页",
     "门卫与 top-k、重排与分组矩阵乘、共享专家、EP 三种通信、EPLB、wave 协调"),
    ("vLLM昇腾算子层.drawio",
     "站11", "每个算子最后由谁执行 · 10 页",
     "融合与下发、矩阵乘三类、小算子分叉、采样的反向选择、TP/EP 两套通信、六层回退、流与同步"),
    ("vLLM组合配置与参数联动.drawio",
     "站8", "同时打开时会怎样 · 10 页",
     "K×桶、显存三链、调度推导链、PD 硬约束、两个 eager、五套配方"),
    ("vLLM故障案例与排查.drawio",
     "站9", "坏的时候长什么样 · 10 页",
     "排障四步法 + 8 个真实复盘 + 十条误导线索"),
]


def plain(fragment: str) -> str:
    """Strip drawio inline html tags, keep text."""
    s = re.sub(r"<br\s*/?>", "\n", fragment)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    return s.strip()


CONCL_KEYS = ("一句话记住", "这一页要记住的", "一句话：", "一句话结论")
ANCHOR_HINT = re.compile(r"\.(py|md):\d+")


def pick_texts(page):
    """Return (title, subtitle, conclusion, [anchor lines]) from one page.

    The ten sets were written at different times, so three phrasings exist for
    the takeaway line; and the oldest set has no takeaway box at all (it uses a
    two-column call-stack layout). Extraction therefore tries several markers
    and simply returns empty strings when nothing matches.
    """
    verts = [v for v in page["verts"].values() if rc.visible(v.id, page["parents"])]
    verts.sort(key=lambda v: (v.y, v.x))
    title = sub = concl = ""
    anchors: list[str] = []
    for v in verts:
        t = plain(v.label)
        if not t:
            continue
        # --- takeaway line ---
        if not concl:
            for key in CONCL_KEYS:
                if key in t:
                    body = t.split(key, 1)[1].lstrip("：:　 ")
                    if len(body) > 12:
                        concl = body.replace("\n", " ")
                    break
        # --- code anchors ---
        if "代码锚点" in t or (len(ANCHOR_HINT.findall(t)) >= 2 and "：" in t):
            body = t.split("\n", 1)[1] if "\n" in t else t
            lines = [x.strip() for x in body.split("\n") if ANCHOR_HINT.search(x)]
            if lines:
                anchors = [x for x in lines if x not in anchors]
        # --- title / subtitle ---
        if v.y < 60 and v.w > 600 and not title and len(t) < 80:
            title = t.replace("\n", " ")
        elif v.y < 125 and not sub and len(t) < 200:
            sub = t.replace("\n", " ")
    if not title and page.get("name"):
        title = page["name"]
    return title, sub, concl, anchors


CSS = """
:root { --ink:#1f2328; --dim:#57606a; --line:#d0d7de; --bg:#ffffff;
        --soft:#f6f8fa; --accent:#0969da; }
* { box-sizing: border-box; }
body { margin:0; padding:0 20px 80px; background:var(--bg); color:var(--ink);
       font:15px/1.7 "Microsoft YaHei","PingFang SC",system-ui,sans-serif; }
nav.top { position:sticky; top:0; z-index:9; background:rgba(255,255,255,.94);
          backdrop-filter:blur(6px); border-bottom:1px solid var(--line);
          padding:10px 0; margin-bottom:24px; font-size:13px; }
nav.top a { color:var(--accent); text-decoration:none; margin-right:14px; }
nav.top a:hover { text-decoration:underline; }
h1 { font-size:24px; margin:18px 0 6px; }
h2 { font-size:19px; margin:34px 0 8px; padding-top:10px;
     border-top:1px solid var(--line); }
.lead { color:var(--dim); margin:0 0 18px; }
figure { margin:0 0 34px; }
figcaption { font-size:14px; color:var(--dim); margin:8px 0 0; }
.svgwrap { overflow-x:auto; border:1px solid var(--line); border-radius:8px;
           background:#fff; padding:6px; }
.svgwrap svg { display:block; height:auto; max-width:100%; }
.ptitle { font-size:16px; font-weight:700; margin:0 0 4px; }
.psub { color:var(--dim); font-size:13px; margin:0 0 10px; }
.concl { background:var(--soft); border-left:4px solid var(--accent);
         padding:10px 14px; margin:12px 0 0; font-size:14px; border-radius:0 6px 6px 0; }
.concl b { color:var(--ink); }
details { margin-top:10px; }
summary { cursor:pointer; color:var(--accent); font-size:13px; }
pre { background:var(--soft); padding:10px 12px; border-radius:6px;
      white-space:pre-wrap; word-break:break-word; font-size:12.5px;
      line-height:1.65; margin:8px 0 0; }
.cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr));
         gap:14px; margin:18px 0 10px; }
.card { border:1px solid var(--line); border-radius:10px; padding:14px 16px;
        text-decoration:none; color:inherit; display:block; }
.card:hover { border-color:var(--accent); background:var(--soft); }
.card .k { color:var(--dim); font-size:12px; letter-spacing:.05em; }
.card .t { font-size:15px; font-weight:700; margin:2px 0 4px; }
.card .d { color:var(--dim); font-size:13px; }
.tag { display:inline-block; font-size:12px; color:var(--dim);
       border:1px solid var(--line); border-radius:99px; padding:1px 9px; margin-left:6px; }
"""

NAV = ('<nav class="top">{links} <a href="index.html">总目录</a>'
       ' <a href="../学习地图.md">学习地图.md</a></nav>')


def page_html(page, idx: int, scale: float) -> str:
    title, sub, concl, anchors = pick_texts(page)
    svg = rc.to_svg(page)
    # scale the root svg element
    svg = svg.replace('width="', f'width="{int(page["w"] * scale)}"', 1)
    svg = svg.replace('height="', f'height="{int(page["h"] * scale)}"', 1)
    out = [f'<figure id="p{idx}">']
    out.append(f'<p class="ptitle">{html.escape(title)}</p>')
    if sub:
        out.append(f'<p class="psub">{html.escape(sub)}</p>')
    out.append(f'<div class="svgwrap">{svg}</div>')
    if concl:
        out.append(f'<div class="concl"><b>一句话结论　</b>{html.escape(concl)}</div>')
    if anchors:
        body = "\n".join(anchors)
        out.append(f'<details><summary>代码锚点（{len(anchors)} 条，可 grep）</summary>'
                   f'<pre>{html.escape(body)}</pre></details>')
    out.append("</figure>")
    return "\n".join(out)


def main() -> int:
    scale = 0.62
    if "--scale" in sys.argv:
        scale = float(sys.argv[sys.argv.index("--scale") + 1])
    os.makedirs(BOOK, exist_ok=True)

    built = []
    for fname, station, label, blurb in SETS:
        path = os.path.join(DRAWIO, fname)
        if not os.path.exists(path):
            print(f"! missing {fname}")
            continue
        pages = rc.load(path)
        slug = re.sub(r"\.drawio$", "", fname)
        slug = slug.replace(" ", "_")
        links = " ".join(f'<a href="#p{i}">{i + 1}</a>' for i in range(len(pages)))
        body = [f'<h1>{station}　{html.escape(label)}</h1>',
                f'<p class="lead">{html.escape(blurb)}</p>',
                f'<p class="lead">本页 {len(pages)} 张图；点击编号跳转：{links}</p>']
        for i, page in enumerate(pages):
            body.append(page_html(page, i, scale))
        doc = (f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
               f'<meta name="viewport" content="width=device-width,initial-scale=1">'
               f'<title>{html.escape(station + " " + label)}</title>'
               f'<style>{CSS}</style></head><body>'
               + NAV.format(links="") + "\n".join(body) + '</body></html>')
        out = os.path.join(BOOK, slug + ".html")
        io.open(out, "w", encoding="utf-8", newline="\n").write(doc)
        built.append((station, label, blurb, slug, len(pages), fname))
        print(f"{station} {label}: {len(pages)} pages -> book/{slug}.html")

    # ---- master index -------------------------------------------------
    total = sum(b[4] for b in built)
    cards = []
    for station, label, blurb, slug, n, fname in built:
        cards.append(
            f'<a class="card" href="{slug}.html">'
            f'<div class="k">{station}<span class="tag">{n} 页</span></div>'
            f'<div class="t">{html.escape(label)}</div>'
            f'<div class="d">{html.escape(blurb)}</div></a>')
    index = (f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
             f'<meta name="viewport" content="width=device-width,initial-scale=1">'
             f'<title>vLLM 与 vLLM-Ascend 图解全集</title>'
             f'<style>{CSS}</style></head><body>' + NAV.format(links="") +
             f'<h1>vLLM 与 vLLM-Ascend 图解全集</h1>'
             f'<p class="lead">十站、{total} 页。顺序不可换：每一站都用到前一站的结论。'
             f'全套的阅读顺序、页间接口与自测题见 '
             f'<a href="../学习地图.md">学习地图.md</a>。</p>'
             f'<div class="cards">' + "\n".join(cards) + '</div>'
             f'<h2>怎么读</h2>'
             f'<p class="lead">① 先读本站的「一句话结论」再细看图；'
             f'② 需要核对代码时展开「代码锚点」，按 文件:行号 grep；'
             f'③ 图与 <code>.drawio</code> 同源（本册由生成器从 .drawio 直接渲染），'
             f'改图后重跑 <code>python tools/makebook.py</code> 即可刷新。</p>'
             f'</body></html>')
    io.open(os.path.join(BOOK, "index.html"), "w", encoding="utf-8",
            newline="\n").write(index)

    # ---- plain index ---------------------------------------------------
    lines = ["# vLLM 与 vLLM-Ascend 图解全集（单卷索引）", "",
             f"共 **{len(built)} 站、{total} 页**。浏览器打开 `index.html` 阅读。", "",
             "| 站 | 材料 | 页数 | 定位 |", "| --- | --- | --- | --- |"]
    for station, label, blurb, slug, n, fname in built:
        lines.append(f"| {station} | [{label}]({slug}.html) | {n} | {blurb} |")
    lines += ["", "每张图下方给出「一句话结论」，折叠区里是可 grep 的「代码锚点」。",
              "", "## 重生成", "", "```bash",
              "cd assets/vllm-ascend/drawio/tools", "python makebook.py",
              "```", "", "本册由 `tools/makebook.py` 从 `.drawio` 文件直接渲染，"
              "与图源同源；改图后重跑即可刷新。", ""]
    io.open(os.path.join(BOOK, "README.md"), "w", encoding="utf-8",
            newline="\n").write("\n".join(lines))
    print(f"index.html + README.md written; total {total} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
