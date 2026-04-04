#!/usr/bin/env python3
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUMMARY_RE = re.compile(r"^(bmpmm|rvv)_(binary|INT2|INT4)_(.+)$")
MODEL_TOTAL_RE = re.compile(r"\bmodel_total\b.*\b(bmpmm_cycles|rvv_cycles)=(\d+)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def read_summary(path: Path):
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            app = row["app"]
            m = SUMMARY_RE.match(app)
            if not m:
                raise ValueError(f"Unexpected app name: {app}")
            impl, precision, model = m.groups()
            log_path = path.parent / Path(row["logfile"]).name if not Path(row["logfile"]).is_absolute() else Path(row["logfile"])
            if not log_path.exists():
                log_path = path.parent / "batch_00" / Path(row["logfile"]).name
            rows.append(
                {
                    "app": app,
                    "impl": impl,
                    "precision": precision,
                    "model": model,
                    "status": row["status"],
                    "duration_sec": int(row["duration_sec"]),
                    "logfile": str(log_path),
                }
            )
    return rows


def extract_model_cycles(log_path: Path):
    text = log_path.read_text(errors="ignore")
    m = MODEL_TOTAL_RE.search(text)
    if not m:
        raise ValueError(f"Missing model_total cycles in {log_path}")
    return int(m.group(2))


def augment_cycles(rows):
    for row in rows:
        row["model_cycles"] = extract_model_cycles(Path(row["logfile"]))
    return rows


def write_csv(path: Path, rows):
    fieldnames = [
        "app",
        "impl",
        "precision",
        "model",
        "status",
        "duration_sec",
        "model_cycles",
        "logfile",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_group_csv(path: Path, rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["impl"], row["precision"])].append(row)

    out_rows = []
    for (impl, precision), items in sorted(groups.items()):
        durations = [r["duration_sec"] for r in items]
        cycles = [r["model_cycles"] for r in items]
        out_rows.append(
            {
                "impl": impl,
                "precision": precision,
                "count": len(items),
                "duration_min_sec": min(durations),
                "duration_avg_sec": round(sum(durations) / len(durations), 2),
                "duration_max_sec": max(durations),
                "cycles_min": min(cycles),
                "cycles_avg": round(sum(cycles) / len(cycles), 2),
                "cycles_max": max(cycles),
            }
        )

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)


def plot_duration_by_app(rows, out_path: Path):
    rows = sorted(rows, key=lambda r: (r["duration_sec"], r["impl"], r["precision"], r["model"]))
    labels = [r["app"] for r in rows]
    values = [r["duration_sec"] for r in rows]
    colors = ["#4C78A8" if r["impl"] == "bmpmm" else "#F58518" for r in rows]

    fig_h = max(10, len(rows) * 0.25)
    plt.figure(figsize=(16, fig_h))
    plt.barh(labels, values, color=colors)
    plt.xlabel("Wallclock Duration (s)")
    plt.ylabel("App")
    plt.title("60-App Run Wallclock Duration")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_cycles_by_app(rows, out_path: Path):
    rows = sorted(rows, key=lambda r: r["model_cycles"])
    labels = [r["app"] for r in rows]
    values = [r["model_cycles"] for r in rows]
    colors = ["#54A24B" if r["impl"] == "bmpmm" else "#E45756" for r in rows]

    fig_h = max(10, len(rows) * 0.25)
    plt.figure(figsize=(16, fig_h))
    plt.barh(labels, values, color=colors)
    plt.xscale("log")
    plt.xlabel("Predicted Model Cycles (log scale)")
    plt.ylabel("App")
    plt.title("60-App Predicted Cycles")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_group_cycles(rows, out_path: Path):
    models = sorted({r["model"] for r in rows})
    combos = [("bmpmm", "binary"), ("bmpmm", "INT2"), ("bmpmm", "INT4"), ("rvv", "binary"), ("rvv", "INT2"), ("rvv", "INT4")]
    combo_labels = [f"{impl}-{prec}" for impl, prec in combos]
    data = {(r["model"], r["impl"], r["precision"]): r["model_cycles"] for r in rows}

    x = list(range(len(models)))
    width = 0.12
    plt.figure(figsize=(18, 7))
    for idx, (impl, prec) in enumerate(combos):
        ys = [data[(m, impl, prec)] for m in models]
        offsets = [v + (idx - 2.5) * width for v in x]
        plt.bar(offsets, ys, width=width, label=f"{impl}-{prec}")

    plt.yscale("log")
    plt.xticks(x, models, rotation=35, ha="right")
    plt.ylabel("Predicted Model Cycles (log scale)")
    plt.title("Predicted Cycles by Model / Implementation / Precision")
    plt.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_markdown(path: Path, rows):
    total = len(rows)
    pass_count = sum(1 for r in rows if r["status"] == "PASS")
    durations = [r["duration_sec"] for r in rows]
    rvv_rows = [r for r in rows if r["impl"] == "rvv"]
    bmpmm_rows = [r for r in rows if r["impl"] == "bmpmm"]

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    lines = []
    lines.append("# 60-App Run Summary")
    lines.append("")
    lines.append(f"- Total apps: {total}")
    lines.append(f"- PASS: {pass_count}")
    lines.append(f"- Duration range: {min(durations)}s to {max(durations)}s")
    lines.append(f"- Average RVV wallclock: {avg([r['duration_sec'] for r in rvv_rows]):.2f}s")
    lines.append(f"- Average BMPMM wallclock: {avg([r['duration_sec'] for r in bmpmm_rows]):.2f}s")
    lines.append("")

    lines.append("## Slowest RVV Apps")
    lines.append("")
    for row in sorted(rvv_rows, key=lambda r: r["duration_sec"], reverse=True)[:10]:
        lines.append(
            f"- {row['app']}: {row['duration_sec']}s, {row['model_cycles']} cycles"
        )
    lines.append("")

    lines.append("## Highest Predicted Cycles")
    lines.append("")
    for row in sorted(rows, key=lambda r: r["model_cycles"], reverse=True)[:10]:
        lines.append(
            f"- {row['app']}: {row['model_cycles']} cycles, wallclock {row['duration_sec']}s"
        )

    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    summary_path = Path(args.summary).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = augment_cycles(read_summary(summary_path))
    write_csv(out_dir / "report_full.csv", rows)
    write_group_csv(out_dir / "report_grouped.csv", rows)
    plot_duration_by_app(rows, out_dir / "duration_by_app.png")
    plot_cycles_by_app(rows, out_dir / "cycles_by_app.png")
    plot_group_cycles(rows, out_dir / "cycles_by_model_impl_precision.png")
    write_markdown(out_dir / "report.md", rows)


if __name__ == "__main__":
    main()
