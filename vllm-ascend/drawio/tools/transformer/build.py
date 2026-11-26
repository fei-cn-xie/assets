"""Assemble the Transformer + model-file explainer: ../Transformer与模型文件结构.drawio

    cd assets/vllm-ascend/drawio/tools
    python transformer/build.py                  # 写回成品文件
    python transformer/build.py t1 t2 t3        # 只构建指定的几页（写 .preview.drawio）

未写好的页会被跳过（方便边写边看）。
"""
from __future__ import annotations

import importlib
import io
import os
import sys

try:                        # Windows 控制台默认 GBK，中文/特殊符号会炸
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

PAGES = [f"t{i}" for i in range(1, 22)]
DEFAULT_OUT = os.path.join(DRAWIO_DIR, "Transformer与模型文件结构.drawio")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    names = args or PAGES
    out = DEFAULT_OUT if not args else os.path.join(DRAWIO_DIR, ".preview.drawio")

    doc = dg.Doc()
    built = []
    for name in names:
        try:
            mod = importlib.import_module(name)
        except ModuleNotFoundError:
            print(f"skip {name} (not written yet)")
            continue
        importlib.reload(mod)
        mod.build(doc)
        built.append(name)

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
    print(f"wrote {out} ({len(doc.pages)} pages: {', '.join(built)}; "
          f"{problems} layout issues)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
