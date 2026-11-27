"""Sanity check for the generated book: every page has an SVG, most have a
conclusion and anchors. Prints a per-set table.

    python tools/checkbook.py
"""
import glob
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BOOK = os.path.join(os.path.dirname(HERE), "book")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

bad = 0
n_svg = n_con = n_anc = 0
print("file".ljust(40), "svg".rjust(4), "concl".rjust(6), "anch".rjust(6))
for path in sorted(glob.glob(os.path.join(BOOK, "*.html"))):
    name = os.path.basename(path)
    if name == "index.html":
        continue
    s = io.open(path, encoding="utf-8").read()
    svg = s.count("<svg")
    con = s.count('class="concl"')
    anc = s.count("<details>")
    n_svg += svg
    n_con += con
    n_anc += anc
    flag = "  <-- NO DIAGRAMS" if svg == 0 else ""
    print(name.ljust(40), str(svg).rjust(4), str(con).rjust(6),
          str(anc).rjust(6), flag)
    bad += 0 if not flag else 1
print(f"totals: svg={n_svg} concl={n_con} anchors={n_anc}; "
      f"files needing attention: {bad}")
print("note: 站3「生命周期」那套图写作最早，本来就没有「一句话」结论框，"
      "concl=0 属正常；站1 只有 3 页带结论框。")
