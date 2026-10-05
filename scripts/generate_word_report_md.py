#!/usr/bin/env python3
"""Generate the two LBT-style Word report sources (Markdown) for the TRAPs project.

    python3 scripts/generate_word_report_md.py            # writes docs/reports/word/*.md and figures/*.jpg

Numbers come from ``outputs/server/{r1_hourly,d1_daily25h}/aggregate_report`` (override with $TRAPS_RESULTS).
Nothing under the existing reports is modified.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from word_md_common import OUT_DIR, SITES, load_all, two_pass  # noqa: E402
from word_md_figures import convert_site_figures, filter_response, pipeline_flowchart  # noqa: E402
from word_md_localize import localize, method_notes, results_notes  # noqa: E402
from word_md_method import build_method  # noqa: E402
from word_md_results import build_results  # noqa: E402


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats, man = load_all()
    p1 = pipeline_flowchart()
    p2 = filter_response()
    figmap = convert_site_figures(SITES)
    figs = {"flow": f"figures/{p1.name}", "resp": f"figures/{p2.name}"}
    (OUT_DIR / "TRAPs_實作方法.md").write_text(method_notes(localize(two_pass(build_method(stats, man, figs)))), encoding="utf-8")
    (OUT_DIR / "TRAPs_執行成果.md").write_text(results_notes(localize(two_pass(build_results(stats, man, figmap)))), encoding="utf-8")
    print("written", OUT_DIR)


if __name__ == "__main__":
    main()
