"""One-command health check for every diagram set.

    cd assets/vllm-ascend/drawio/tools
    python verify_all.py            # 结构 + 渲染 + 配色，全部 12 套
    python verify_all.py --quick    # 只做结构与配色（跳过逐页渲染）

Exit code 0 means every set passed every check.
"""
from __future__ import annotations

import glob
import io
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DRAWIO = os.path.dirname(HERE)
PY = sys.executable
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

QUICK = "--quick" in sys.argv
files = sorted(glob.glob(os.path.join(DRAWIO, "*.drawio")))
if not files:
    print("no .drawio found")
    raise SystemExit(1)


def run(script: str, args: list[str]) -> tuple[int, str]:
    p = subprocess.run([PY, os.path.join(HERE, script)] + args,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    tail = (p.stdout or "").strip().splitlines()[-1:] or [""]
    return p.returncode, tail[0]


hdr = f"{'set':34s} {'pages':>5s} {'struct':>6s} {'render':>6s} {'color':>6s}"
print(hdr)
print("-" * len(hdr))
bad = 0
total_pages = 0
for path in files:
    name = os.path.basename(path)
    # page count
    raw = io.open(path, encoding="utf-8").read()
    pages = raw.count("<diagram")
    total_pages += pages
    rc1, _ = run("checkdrawio.py", [path])
    if QUICK:
        rc2, _ = (0, "skipped")
    else:
        rc2, _ = run("render_check.py",
                     [path, "-o", os.path.join(DRAWIO, "..", "..", "..",
                                               ".tmp", "verify",
                                               name.replace(".drawio", "")),
                      "--scale", "0.5"])
    rc3, _ = run("color_audit.py", [path])
    def mark(rc):
        return "OK" if rc == 0 else "FAIL"
    print(f"{name[:34]:34s} {pages:5d} {mark(rc1):>6s} {mark(rc2):>6s} {mark(rc3):>6s}")
    bad += (rc1 != 0) + (rc2 != 0) + (rc3 != 0)

print("-" * len(hdr))
print(f"sets={len(files)}  pages={total_pages}  failing checks={bad}")
print("OK: all sets passed" if bad == 0 else "SOME CHECKS FAILED")
raise SystemExit(0 if bad == 0 else 1)
