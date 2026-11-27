"""MTP / 投机解码：draft 是什么、为什么能加速、vLLM 里怎么实现。

这一套讲「用小模型/多头猜几个 token，再让大模型一次验证」的完整体系：
draft 从哪来 → 怎么验证 → 被拒了怎么办 → 与图捕获/显存/调度的耦合。

    cd assets/vllm-ascend/drawio/tools
    python intro/build.py
"""
from __future__ import annotations

import importlib
import io
import os
import sys

try:
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

PAGES = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"]
DEFAULT_OUT = os.path.join(DRAWIO_DIR, "vLLM的MTP与投机解码.drawio")


def main() -> int:
    doc = dg.Doc()
    for name in PAGES:
        mod = importlib.import_module(name)
        importlib.reload(mod)
        mod.build(doc)
    out = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT

    text = fmt.normalized(doc.xml())

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
