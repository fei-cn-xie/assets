"""Print page names / cell counts of a .drawio (avoids PowerShell quoting traps).

    python listpages.py <file.drawio> [more.drawio ...]
"""
import io
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

for path in sys.argv[1:]:
    src = io.open(path, encoding="utf-8").read()
    names = re.findall(r'<diagram[^>]*?name="([^"]*)"', src)
    print(f"== {os.path.basename(path)}  ({len(names)} pages, "
          f"{src.count('<mxCell')} cells)")
    for i, n in enumerate(names):
        print(f"   [{i:02d}] {n}")
