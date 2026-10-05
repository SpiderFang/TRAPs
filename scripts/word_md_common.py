"""Shared helpers for generating the LBT-style Word report sources (Markdown).

Used by ``generate_word_report_md.py``.  Nothing here touches the repository's
existing reports; results are read from ``outputs/server`` (or ``$TRAPS_RESULTS``).
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_RESULTS = ROOT / "outputs" / "server"
if not _DEFAULT_RESULTS.exists() and Path("/home/claude/traps_results/outputs/server").exists():
    _DEFAULT_RESULTS = Path("/home/claude/traps_results/outputs/server")  # sandbox copy of the Mac outputs/
RESULTS = Path(os.environ.get("TRAPS_RESULTS", _DEFAULT_RESULTS))
OUT_DIR = ROOT / "docs" / "reports" / "word"
FIG_DIR = OUT_DIR / "figures"

RUNS = {"hourly": "r1_hourly", "daily": "d1_daily25h"}
SITES = [
    ("gongliao", "貢寮", "A"),
    ("guishan", "龜山島西側", "A"),
    ("hsinchu", "新竹外海", "B"),
    ("nanwan", "南灣", "C"),
    ("lienchiang", "連江", "D"),
]
SEASONS = ["DJF", "MAM", "JJA", "SON"]
INDENT = "　　"
MINUS = "−"


# --------------------------------------------------------------------------- data
def load_all():
    stats, man = {}, {}
    for key, run in RUNS.items():
        base = RESULTS / run / "aggregate_report"
        j = json.loads((base / "site_statistics.json").read_text(encoding="utf-8"))
        d = {}
        for r in j["records"]:
            d.setdefault(r["site_id"], {})[r["stratum"]] = r
        stats[key] = d
        man[key] = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
        man[key]["_track_core_filter_json"] = j["track_core_filter"]
    return stats, man


# --------------------------------------------------------------------------- formats
def g(x, n=4):
    """n significant digits, never scientific notation, trailing zeros trimmed."""
    if x is None:
        return "—"
    x = float(x)
    if x == 0 or abs(x) < 1e-9:
        return "0"
    s = format(float(f"{x:.{n}g}"), "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s.replace("-", MINUS)


def pct(x, n=4):
    return "—" if x is None else g(float(x) * 100, n) + "%"


def num(n):
    return f"{int(round(float(n))):,}"


def sci(x, n=4):
    """−1.719×10⁻⁴ style."""
    m, e = f"{float(x):.{n - 1}e}".split("e")
    m = m.rstrip("0").rstrip(".") if "." in m else m
    e = int(e)
    sup = str(abs(e)).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    sign = "⁻" if e < 0 else ""
    return f"{m}×10{sign}{sup}".replace("-", MINUS)


def ci(pair, n=4):
    """Confidence interval, always two decimals: 9.96–10.10%."""
    return f"{float(pair[0]) * 100:.2f}–{float(pair[1]) * 100:.2f}%"


def cip(pair):
    return "（" + ci(pair) + "）"


def tex_sci(x, n=4):
    m, e = f"{float(x):.{n - 1}e}".split("e")
    m = m.rstrip("0").rstrip(".") if "." in m else m
    return f"{m}\\times 10^{{{int(e)}}}"


def ratio(x):
    return "—" if x is None else g(x, 3) + "×"


# --------------------------------------------------------------------------- Doc builder
class Doc:
    """Collects Markdown lines in the LBT dialect; run twice so that forward
    references (tables, figures, equations) resolve on the second pass."""

    def __init__(self, tprefix, fprefix, refs=None):
        self.t, self.f = tprefix, fprefix
        self.out: list[str] = []
        self.refs = refs if refs is not None else {}
        self.new = {}
        self.nt = self.nf = self.ne = 0

    # references -----------------------------------------------------------
    def T(self, key):
        return self.refs.get(("t", key), "表?-?")

    def F(self, key):
        return self.refs.get(("f", key), "圖?-?")

    def E(self, key):
        return "(%s)" % self.refs.get(("e", key), "?")

    # blocks ---------------------------------------------------------------
    def raw(self, s=""):
        self.out.append(s)
        self.out.append("")

    def h(self, level, text):
        self.raw("#" * level + " " + text)

    def p(self, text):
        self.raw(INDENT + text)

    def plain(self, text):
        self.raw(text)

    def note(self, text):
        self.raw("〔編輯說明：" + text + "〕")

    def eq(self, key, tex):
        self.ne += 1
        self.new[("e", key)] = self.ne
        self.raw("$$\n" + tex.strip() + r" \qquad (%d)" % self.ne + "\n$$")

    def table(self, key, title, header, rows, source):
        self.nt += 1
        label = f"{self.t}-{self.nt}"
        self.new[("t", key)] = label
        self.raw(f"**{label}  {title}**")
        lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for r in rows:
            assert len(r) == len(header), (title, r)
            lines.append("| " + " | ".join(str(c).replace("|", "／") for c in r) + " |")
        self.out.extend(lines)
        self.out.append("")
        self.raw("資料來源：" + source)

    def fig(self, key, title, path, width="6.3in"):
        self.nf += 1
        label = f"{self.f}-{self.nf}"
        self.new[("f", key)] = label
        self.raw(f"![]({path}){{width={width}}}")
        self.raw(f"**{label}  {title}**")

    def text(self):
        return "\n".join(self.out).rstrip() + "\n"


def two_pass(build):
    """build(doc) -> None.  Returns the final text."""
    d1 = Doc(*build.prefixes)
    build(d1)
    d2 = Doc(*build.prefixes, refs=d1.new)
    build(d2)
    assert d2.new == d1.new
    return d2.text()


def tracks_n(rec, frac_key, n_key="track_n_uncensored"):
    return int(round(rec[frac_key] * rec[n_key]))
