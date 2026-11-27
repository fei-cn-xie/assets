"""过渡图：从「训练」到「推理系统」——把 DL 基础与 vLLM 的机制接上。

这一套不是讲某个功能，而是补「为什么需要 vLLM 那些机制」的前置推理链：
训练产出权重 → 推理两个阶段 → 算术强度决定瓶颈 → 于是要 batching / KV 复用 /
并行 / 图捕获 → 这些需求一一对应到 vLLM 的组件。

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

PAGES = ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "k9", "k10"]
DEFAULT_OUT = os.path.join(DRAWIO_DIR, "vLLM故障案例与排查.drawio")


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
