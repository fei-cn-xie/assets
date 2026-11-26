"""Assemble the graph-capture explainer set: ../vLLM图捕获.drawio

    cd assets/vllm-ascend/drawio/tools
    python graph/build.py

读者定位：懂 Transformer 与 Python、但没接触过 CUDA/ACL Graph 的人。
每页编排：一句话结论 → 图 → 硬约束/代价 → 代码锚点。
"""
from __future__ import annotations

import importlib
import io
import os
import sys

try:                      # Windows 控制台默认 GBK，打印 ⑪ 这类字符会炸
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
DRAWIO_DIR = os.path.dirname(TOOLS)
for p in (TOOLS, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import dg                      # noqa: E402
import format as fmt           # noqa: E402

PAGES = ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10"]
DEFAULT_OUT = os.path.join(DRAWIO_DIR, "vLLM图捕获.drawio")


def main() -> int:
    doc = dg.Doc()
    for name in PAGES:
        mod = importlib.import_module(name)
        importlib.reload(mod)
        mod.build(doc)
    out = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT

    text = fmt.normalized(doc.xml())      # 先完整生成（懒构建在这里落地）

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
