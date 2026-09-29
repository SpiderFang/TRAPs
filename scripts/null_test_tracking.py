#!/usr/bin/env python3
"""比較真實與時間置換（null model）追蹤結果的 track 壽命分布。

輸入為多個 ``LABEL=TRACK_ROOT``，每個 TRACK_ROOT 下含 ``<domain>/`` 子目錄（``track-period``
輸出，可帶 ``--shuffle-seed``）。輸出 ``comparison.csv``、``comparison.png`` 並列印
「coherence ratio」＝ 真實 fraction(壽命 >= 3 步) / 置換組（跨 seed）平均 fraction(>= 3 步)。
比值明顯大於 1 才表示 track 具備超越隨機串接的時間連續性。

圖中僅使用英文標籤與預設 sans 字型，避免 CJK 字型相依。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CSV_FIELDS = (
    "run", "domain", "gate_core_m", "gate_curve_m", "shuffled", "seed", "n_tracks", "n_uncensored",
    "frac_lifetime_1", "median_steps", "p90_steps", "frac_ge3", "frac_ge7", "mean_steps", "coherence_ratio",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _steps_from_hist(hist: dict[str, Any]) -> np.ndarray:
    """把 lifetime_hist.json 的計數展開成每條未截尾 track 的壽命步數。"""

    return np.concatenate(
        [np.full(int(count), int(step), dtype=np.int64) for step, count in hist["counts"].items()]
        + [np.zeros(0, dtype=np.int64)]
    )


def load_run(label: str, root: Path) -> list[dict[str, Any]]:
    """讀取一個 run 下所有 domain，回傳含統計與壽命陣列的紀錄。"""

    records: list[dict[str, Any]] = []
    for domain_dir in sorted(p for p in root.iterdir() if (p / "lifetime_hist.json").is_file()):
        summary = _read_json(domain_dir / "tracking_summary.json")
        hist = _read_json(domain_dir / "lifetime_hist.json")
        manifest = _read_json(domain_dir / "manifest.json")
        cfg = manifest["tracking_config"]
        null = manifest.get("null_model")
        steps = _steps_from_hist(hist)
        n = int(steps.size)

        def frac(mask: np.ndarray) -> float:
            return float(mask.mean()) if n else float("nan")

        records.append(
            {
                "run": label,
                "domain": domain_dir.name,
                "gate_core_m": float(cfg["max_core_distance_m"]),
                "gate_curve_m": float(cfg["max_curve_distance_m"]),
                "shuffled": null is not None,
                "seed": "" if null is None else int(null["seed"]),
                "n_tracks": int(summary["track_count"]),
                "n_uncensored": n,
                "frac_lifetime_1": frac(steps == 1),
                "median_steps": float(np.median(steps)) if n else float("nan"),
                "p90_steps": float(np.quantile(steps, 0.9)) if n else float("nan"),
                "frac_ge3": frac(steps >= 3),
                "frac_ge7": frac(steps >= 7),
                "mean_steps": float(steps.mean()) if n else float("nan"),
                "coherence_ratio": "",
                "_steps": steps,
            }
        )
    if not records:
        raise SystemExit(f"{root} 下找不到含 lifetime_hist.json 的 domain 子目錄")
    return records


def add_coherence_ratio(records: list[dict[str, Any]]) -> dict[tuple[str, float, float], float]:
    """對每個 (domain, gate) 計算真實 / 置換平均的 fraction(>=3 步)，填入真實列。"""

    shuffled: dict[tuple[str, float, float], list[float]] = defaultdict(list)
    for r in records:
        if r["shuffled"]:
            shuffled[(r["domain"], r["gate_core_m"], r["gate_curve_m"])].append(r["frac_ge3"])
    ratios: dict[tuple[str, float, float], float] = {}
    for r in records:
        key = (r["domain"], r["gate_core_m"], r["gate_curve_m"])
        if r["shuffled"] or key not in shuffled:
            continue
        null_mean = float(np.nanmean(shuffled[key]))
        ratio = r["frac_ge3"] / null_mean if null_mean > 0 else float("inf")
        r["coherence_ratio"] = ratio
        ratios[key] = ratio
    return ratios


def plot_ccdf(records: list[dict[str, Any]], path: Path) -> None:
    domains = sorted({r["domain"] for r in records})
    gates = sorted({(r["gate_core_m"], r["gate_curve_m"]) for r in records})
    colors = {g: plt.get_cmap("tab10")(i % 10) for i, g in enumerate(gates)}
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"]}):
        fig, axes = plt.subplots(1, len(domains), figsize=(4.6 * len(domains), 3.8), squeeze=False, sharey=True)
        for ax, domain in zip(axes[0], domains):
            for r in (x for x in records if x["domain"] == domain and x["n_uncensored"] > 0):
                steps = np.sort(r["_steps"])
                ks = np.arange(1, int(steps[-1]) + 1)
                ccdf = np.array([(steps >= k).mean() for k in ks])
                gate = (r["gate_core_m"], r["gate_curve_m"])
                ax.step(
                    ks, ccdf, where="post", color=colors[gate], linestyle="--" if r["shuffled"] else "-",
                    label=f"{r['run']} ({'shuffled' if r['shuffled'] else 'real'}, "
                    f"{r['gate_core_m']:g}/{r['gate_curve_m']:g} m)",
                )
            ax.set_yscale("log")
            ax.set_xlabel("lifetime (nominal steps)")
            ax.set_title(domain, fontsize=10)
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=7)
        axes[0][0].set_ylabel("P(lifetime >= k), uncensored")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True, metavar="LABEL=TRACK_ROOT")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    records: list[dict[str, Any]] = []
    for item in args.runs:
        label, sep, root = item.partition("=")
        if not sep or not label or not root:
            parser.error(f"--runs 需為 LABEL=TRACK_ROOT: {item!r}")
        records.extend(load_run(label, Path(root)))
    ratios = add_coherence_ratio(records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    plot_ccdf(records, args.output_dir / "comparison.png")

    if ratios:
        for (domain, core, curve), ratio in sorted(ratios.items()):
            print(f"coherence ratio {domain} gate {core:g}/{curve:g} m: {ratio:.3f}  (frac>=3 real / shuffled mean)")
    else:
        print("no shuffled runs given: coherence ratio not computed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
