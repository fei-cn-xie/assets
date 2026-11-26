"""Assemble the Mooncake explainer set: ../Mooncake原理与实现.drawio

    cd assets/vllm-ascend/drawio/tools
    python mooncake/build.py                 # 写回 ../Mooncake原理与实现.drawio

读者定位：大一新生（会 Python 基础、学过 Transformer 的 attention）。
因此每页都遵循同一条编排：**先一句话结论 → 再大白话类比 → 最后才落到代码锚点**。
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

PAGES = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m10"]
DEFAULT_OUT = os.path.join(DRAWIO_DIR, "Mooncake原理与实现.drawio")


def main() -> int:
    doc = dg.Doc()
    for name in PAGES:
        mod = importlib.import_module(name)
        importlib.reload(mod)
        mod.build(doc)
    out = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT

    text = fmt.normalized(doc.xml())     # 先完整生成（懒构建在这里落地）

    problems = 0
    for p in doc.pages:
        issues = dg.validate(p)
        print(f"[{p.name}] cells={len(p.cells)} layout_issues={len(issues)}")
        for i in issues[:12]:
            print("   -", i)
        problems += len(issues)

    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"wrote {out} ({len(doc.pages)} pages, {problems} layout issues)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
