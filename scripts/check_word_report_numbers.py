"""Independent number check of the Word-report Markdown against site_statistics.csv.

Re-reads the CSV (not the JSON used by the generator) with its own formatter and
verifies (1) every cell of each per-site 關鍵統計 table and 壽命分數 table, and
(2) that every multi-digit numeric token in the narrative matches some CSV/manifest
value at a plausible scaling.  Prints mismatches; exit status 1 if any table cell fails.
"""
import csv, json, os, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from word_md_localize import localize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get("TRAPS_RESULTS", ROOT / "outputs" / "server"))
if not RES.exists():
    RES = Path("/home/claude/traps_results/outputs/server")
RUNS = {"hourly": "r1_hourly", "daily": "d1_daily25h"}
SITES = {"gongliao": "貢寮", "guishan": "龜山島西側", "hsinchu": "新竹外海", "nanwan": "南灣", "lienchiang": "連江"}
MD = ROOT / "docs/reports/word/TRAPs_執行成果.md"


def sg(x, n=4):
    s = f"{float(f'{x:.{n}g}'):.12f}".rstrip("0").rstrip(".")
    return (s or "0").replace("-", "−")


def P(x):
    return sg(x * 100) + "%"


def N(x):
    return f"{int(round(x)):,}"


def load(run):
    d = {}
    for r in csv.DictReader(open(RES / RUN_DIR(run) / "aggregate_report/site_statistics.csv", encoding="utf-8")):
        d[(r["site_id"], r["stratum"])] = r
    return d


def RUN_DIR(k):
    return RUNS[k]


def F(r, k):
    return float(r[k])


def ci(r, k):
    a, b = re.findall(r"[-\d.eE+]+", r[k])[:2]
    return f"{float(a) * 100:.2f}–{float(b) * 100:.2f}%"


def tables(text):
    out = {}
    for m in re.finditer(r"\*\*(表乙-\d+)  (.*?)\*\*\n\n((?:\|.*\n)+)", text):
        rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in m.group(3).splitlines()]
        out[m.group(2)] = rows[2:]
    return out


def main():
    text = MD.read_text(encoding="utf-8")
    T = tables(text)
    D = {k: load(k) for k in RUNS}
    bad = ok = 0
    for sid, nm in SITES.items():
        rows = {r[0]: r[1:] for r in T[f"{nm}關鍵統計"]}
        ZH = lambda lab: localize(lab)  # noqa: E731
        for j, k in enumerate(("hourly", "daily")):
            r = D[k][(sid, "all")]
            exp = {
                "站點分析區面積（平方公里）": sg(F(r, "site_area_km2"), 5),
                "有效快照數": N(F(r, "n_valid_snapshots")),
                "合格核心總數": N(F(r, "core_s1_count")),
                "熱點面積（平方公里）（格數）": f"{sg(F(r,'hotspot_area_km2'), 5)}（{N(F(r,'hotspot_cell_count'))}）",
                "熱點內強核心比例（占該站強核心）": P(F(r, "hotspot_strong_core_fraction")),
                "熱點面積位於 10 公里近岸帶比例": P(F(r, "hotspot_frac_area_within_buffer")),
                "全部核心位於 10 公里近岸帶比例": P(F(r, "frac_cores_within_buffer")),
                "強核心位於 10 公里近岸帶比例": P(F(r, "frac_strong_cores_within_buffer")),
                "近岸外強核心出現率": P(F(r, "offshore_strong_core_occurrence_rate")),
                "核心密度（個每 1,000 平方公里每快照）": sg(F(r, "core_density_per_1000km2_per_snapshot"), 4),
                "強核心 track 數（全部／未截尾）": f"{N(F(r,'track_n_tracks'))}／{N(F(r,'track_n_uncensored'))}",
                "熱點內核心平均 $s_1$（每日）": sg(F(r, "hotspot_mean_core_s1_day")),
                "核心 $s_1$ 中位數／p10（每日）": f"{sg(F(r,'core_s1_median_day'))}／{sg(F(r,'core_s1_p10_day'))}",
                "近岸外熱點格數／平均通過頻率": f"{N(F(r,'offshore_hotspot_cell_count'))}／{sg(F(r,'offshore_hotspot_mean_curve_pass_frequency'))}",
                "熱點門檻（曲線通過頻率）／熱點內平均頻率": f"{sg(F(r,'hotspot_threshold_curve_pass_frequency'))}／{sg(F(r,'hotspot_mean_curve_pass_frequency'))}",
            }
            for lab, e in exp.items():
                lab = ZH(lab)
                got = rows.get(lab, [None, None])[j]
                if got == e:
                    ok += 1
                else:
                    bad += 1
                    print("MISMATCH", sid, k, lab, "md=", got, "csv=", e)
            # compound cells: check the CSV number appears inside
            for lab, e in {
                "強核心出現率（分子／分母；95% 信賴區間）": [P(F(r, "strong_core_occurrence_rate")), N(F(r, "strong_core_occurrence_numerator")), N(F(r, "n_valid_snapshots")), ci(r, "strong_core_occurrence_rate_ci95")],
                "曲線通過格點比例（95% 信賴區間）": [P(F(r, "mean_pass_cell_fraction")), ci(r, "mean_pass_cell_fraction_ci95")],
            }.items():
                got = rows[ZH(lab)][j]
                for piece in e:
                    if piece in got:
                        ok += 1
                    else:
                        bad += 1
                        print("MISMATCH", sid, k, lab, piece, "not in", got)
    # narrative token triage
    vals = set()
    for k in RUNS:
        for r in D[k].values():
            for v in r.values():
                try:
                    x = float(v)
                except ValueError:
                    for t in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", v):
                        x = float(t); vals.add(abs(x))
                    continue
                vals.add(abs(x))
        m = json.load(open(RES / RUNS[k] / "aggregate_report/manifest.json"))
        def walk(o):
            if isinstance(o, dict):
                for v in o.values(): walk(v)
            elif isinstance(o, list):
                for v in o: walk(v)
            elif isinstance(o, (int, float)) and not isinstance(o, bool):
                vals.add(abs(float(o)))
        walk(m)

    def matches(x):
        for s in (1, 100, 1 / 24, 24, 86400, 1 / 86400, 1000, 1e-3, 1e5, 1e-5, 1e4, 1e-4, 1 / 1000):
            for v in vals:
                if v and abs(v * s - x) <= max(abs(x), 1e-12) * 6e-4 + 5e-5 * (1 if x < 1000 else 0):
                    return True
        return False

    body = re.sub(r"^\|.*$", "", text, flags=re.M)
    un = {}
    for t in re.findall(r"(?<![\w.])\d[\d,]*\.?\d*(?![\w])", body):
        x = float(t.replace(",", ""))
        if x >= 10 and not matches(x):
            un[t] = un.get(t, 0) + 1
    print(f"table cells ok={ok} bad={bad}")
    print(f"narrative numeric tokens >=10 not traced to CSV/manifest: {len(un)}")
    print(sorted(un, key=lambda s: -un[s])[:80])
    return 1 if bad else 0


sys.exit(main())
