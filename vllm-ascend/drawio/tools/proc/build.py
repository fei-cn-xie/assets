"""Assemble the process/thread explainer set: ../vLLM进程与线程.drawio

    cd assets/vllm-ascend/drawio/tools
    python proc/build.py

读者定位：会 Python、想看明白「vLLM 到底起了几个进程、每个进程里有什么线程、
它们怎么通信、参数怎么影响进程数」的人。每页仍遵循：一句话结论 → 图 → 代码锚点。
"""
from __future__ import annotations

import importlib
import io
import os
import sys

try:                      # Windows 控制台默认 GBK，打印 ⑪ 这类字符会炸，强制 UTF-8
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

PAGES = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10",
         "t11", "t12", "t13", "t14", "t15", "t16"]
DEFAULT_OUT = os.path.join(DRAWIO_DIR, "vLLM进程与线程.drawio")


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
