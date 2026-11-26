"""Assemble the pages into vLLM生命周期-端到端.drawio.

    cd assets/vllm-ascend/drawio/tools
    python pages/build.py                  # 写回 ../vLLM生命周期-端到端.drawio
    python pages/build.py /tmp/out.drawio  # 写到别处

输出格式与入库文件完全一致（多行、LF、无 BOM），所以重建后 git diff 应当是空的；
一旦出现 diff，说明页面代码与该 .drawio 已经不同步 —— 先弄清谁才是最新，
不要直接把生成结果覆盖上去。
"""
from __future__ import annotations

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
DRAWIO_DIR = os.path.dirname(TOOLS)
for p in (TOOLS, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import dg                      # noqa: E402
import format as fmt           # noqa: E402

PAGES = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9"]
DEFAULT_OUT = os.path.join(DRAWIO_DIR, "vLLM生命周期-端到端.drawio")


def main() -> int:
    doc = dg.Doc()
    for name in PAGES:
        mod = importlib.import_module(name)
        importlib.reload(mod)
        mod.build(doc)
    out = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(fmt.normalized(doc.xml()))

    problems = 0
    for p in doc.pages:
        issues = dg.validate(p)
        print(f"[{p.name}] cells={len(p.cells)} layout_issues={len(issues)}")
        for i in issues[:20]:
            print("   -", i)
        problems += len(issues)
    print(f"wrote {out} ({len(doc.pages)} pages, {problems} layout issues)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
