#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Embed the generated SVG diagrams into the architecture markdown.

Idempotent: every known image line is first removed, then re-inserted directly
after its section heading. Mermaid source blocks are folded into collapsed
<details> sections.

Run:  python -X utf8 embed-diagrams.py
Always reads/writes UTF-8 explicitly (never rely on the console codepage).
"""

from __future__ import annotations

import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(HERE, "vllm-main-architecture.md")
IMG_PREFIX = "./img/"

# heading prefix -> (image file name, add a "---" separator after the image)
PLACEMENTS = [
    ("## 1. 一图看懂：全局分层架构",
     "architecture-overview.svg", "vLLM main 分支全局分层架构", True),
    ("### 1.3 一次生成请求的端到端流程",
     "request-flow.svg", "一次生成请求的端到端流程", True),
    ("## 2. 进程、线程与通信拓扑",
     "process-topology.svg", "vLLM 进程、线程与通信拓扑", True),
    ("### 7.2 KV 缓存的三层结构",
     "kv-cache-stack.svg", "KV cache 三层结构与卸载机制", False),
    ("### 8.1 执行器选择",
     "executor-topology.svg", "执行器拓扑与默认选择规则", False),
    ("### 8.2 Worker 生命周期",
     "worker-lifecycle.svg", "Worker 生命周期", False),
    ("### 10.3 一次 attention 的数据流",
     "attention-dataflow.svg", "一次 attention 的数据流", False),
    ("### 11.4 结构化输出",
     "bitmask-flow.svg", "结构化输出 bitmask 链路", False),
    ("### 12.1 总管线",
     "input-pipeline.svg", "输入侧总管线", False),
    ("### 14.1 并行维度总览",
     "parallelism.svg", "并行维度：每个维度切分什么", False),
    ("### 16.2 指标通路",
     "metrics-path.svg", "指标通路", False),
]

# Matches both "./img/x.svg" and any legacy "./x.svg" reference.
IMG_LINE_RE = re.compile(r"^!\[[^\]]*\]\(\./(?:img/)?[^/)]+\.svg\)$")

MERMAID_RE = re.compile(r"```mermaid\n(.*?)\n```", re.S)
WRAP_RE = re.compile(
    r"<details>\n"
    r"<summary>Mermaid 源码（可编辑，需支持 mermaid 的渲染器）</summary>\n\n"
    r"(```mermaid\n.*?\n```)\n\n</details>",
    re.S,
)


def main() -> None:
    with open(DOC, "r", encoding="utf-8") as fh:
        text = fh.read()

    # 1) drop every image line (idempotency; also clears legacy non-img/ paths)
    lines = [ln for ln in text.split("\n") if not IMG_LINE_RE.match(ln.strip())]

    # 2) insert each image right after its heading
    inserted = 0
    for heading, fname, alt, sep in PLACEMENTS:
        idx = next((i for i, ln in enumerate(lines) if ln.startswith(heading)), None)
        if idx is None:
            print(f"  !! heading not found: {heading}")
            continue
        img = f"![{alt}]({IMG_PREFIX}{fname})"
        block = [img, ""]
        if sep:
            block += ["---", ""]
        # keep exactly one blank line between heading and image
        insert_at = idx + 1
        while insert_at < len(lines) and lines[insert_at].strip() == "":
            lines.pop(insert_at)
        lines[insert_at:insert_at] = [""] + block
        inserted += 1

    text = "\n".join(lines)

    # 3) fold mermaid source into collapsed details (unwrap first => idempotent)
    text = WRAP_RE.sub(r"\1", text)
    folded = 0

    def fold(match: re.Match[str]) -> str:
        nonlocal folded
        folded += 1
        return (
            "<details>\n"
            "<summary>Mermaid 源码（可编辑，需支持 mermaid 的渲染器）</summary>\n\n"
            "```mermaid\n" + match.group(1) + "\n```\n\n</details>"
        )

    text = MERMAID_RE.sub(fold, text)

    with open(DOC, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(f"placed {inserted} diagram(s), folded {folded} mermaid block(s)")


if __name__ == "__main__":
    main()
