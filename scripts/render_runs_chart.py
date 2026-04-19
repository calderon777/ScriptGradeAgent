import argparse
import csv
import html
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ROOT_DIR / "output" / "end_to_end_suite_full_ec3040_2026-04-19_v2" / "runs.csv"
DEFAULT_OUTPUT = ROOT_DIR / "output" / "end_to_end_suite_full_ec3040_2026-04-19_v2" / "human_vs_model_sorted.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a sorted human-vs-model totals chart from benchmark runs.csv."
    )
    parser.add_argument("--runs", default=str(DEFAULT_RUNS), help="Input runs.csv path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output HTML path.")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader]
    rows.sort(key=lambda row: (float(row["human_total"]), row["participant_id"]))
    return rows


def scale_x(index: int, total: int, left: float, width: float) -> float:
    if total <= 1:
        return left + width / 2
    return left + (index * width / (total - 1))


def scale_y(value: float, min_value: float, max_value: float, top: float, height: float) -> float:
    if max_value <= min_value:
        return top + height / 2
    ratio = (value - min_value) / (max_value - min_value)
    return top + height - (ratio * height)


def build_html(rows: list[dict[str, str]]) -> str:
    width = 1200
    height = 700
    margin_left = 80
    margin_right = 24
    margin_top = 48
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    human_values = [float(row["human_total"]) for row in rows]
    model_values = [float(row["model_total"]) for row in rows]
    min_value = min(human_values + model_values + [0.0])
    max_value = max(human_values + model_values) + 2.0

    tick_step = 10
    tick_max = int(((max_value + tick_step - 1) // tick_step) * tick_step)
    ticks = list(range(0, max(tick_max, tick_step) + 1, tick_step))
    human_points: list[str] = []
    blue_dots: list[str] = []
    labels_rows: list[str] = []

    for index, row in enumerate(rows):
        x = scale_x(index, len(rows), margin_left, plot_width)
        human_total = float(row["human_total"])
        model_total = float(row["model_total"])
        y_human = scale_y(human_total, min_value, max_value, margin_top, plot_height)
        y_model = scale_y(model_total, min_value, max_value, margin_top, plot_height)
        human_points.append(f"{x:.2f},{y_human:.2f}")
        title = html.escape(
            f"{row['participant_id']} | {Path(row['path']).name} | human={human_total:.1f} | model={model_total:.1f}"
        )
        blue_dots.append(
            f"<circle cx=\"{x:.2f}\" cy=\"{y_model:.2f}\" r=\"4\" fill=\"#2563eb\">"
            f"<title>{title}</title></circle>"
        )
        labels_rows.append(
            "<tr>"
            f"<td>{index + 1}</td>"
            f"<td>{html.escape(row['participant_id'])}</td>"
            f"<td>{html.escape(Path(row['path']).name)}</td>"
            f"<td>{human_total:.1f}</td>"
            f"<td>{model_total:.1f}</td>"
            "</tr>"
        )

    grid_lines: list[str] = []
    for tick in ticks:
        y = scale_y(float(tick), min_value, max_value, margin_top, plot_height)
        grid_lines.append(
            f"<line x1=\"{margin_left}\" y1=\"{y:.2f}\" x2=\"{width - margin_right}\" y2=\"{y:.2f}\" "
            "stroke=\"#d1d5db\" stroke-width=\"1\" />"
        )
        grid_lines.append(
            f"<text x=\"{margin_left - 12}\" y=\"{y + 4:.2f}\" text-anchor=\"end\" fill=\"#374151\" font-size=\"12\">{tick}</text>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Human vs Model Totals</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      color: #111827;
      background: #ffffff;
    }}
    h1, p {{
      margin: 0 0 12px 0;
    }}
    .legend {{
      display: flex;
      gap: 20px;
      margin: 8px 0 16px 0;
      font-size: 14px;
    }}
    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .swatch {{
      width: 14px;
      height: 14px;
      display: inline-block;
    }}
    .chart-wrap {{
      border: 1px solid #d1d5db;
      padding: 12px 12px 4px 12px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin-top: 20px;
      font-size: 13px;
    }}
    th, td {{
      border: 1px solid #d1d5db;
      padding: 6px 8px;
      text-align: left;
    }}
    th {{
      background: #f3f4f6;
    }}
  </style>
</head>
<body>
  <h1>Sorted Human Marks vs Model Totals</h1>
  <p>Red line: human totals sorted low to high. Blue dots: model totals for the same submissions.</p>
  <div class="legend">
    <span><span class="swatch" style="background:#dc2626;"></span>Human totals</span>
    <span><span class="swatch" style="background:#2563eb; border-radius:50%;"></span>Model totals</span>
  </div>
  <div class="chart-wrap">
    <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Human versus model totals chart">
      <rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" fill="#ffffff" stroke="#9ca3af" stroke-width="1" />
      {''.join(grid_lines)}
      <line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1.2" />
      <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1.2" />
      <polyline points="{' '.join(human_points)}" fill="none" stroke="#dc2626" stroke-width="2.5" />
      {''.join(blue_dots)}
      <text x="{width / 2:.2f}" y="{height - 18}" text-anchor="middle" fill="#374151" font-size="13">Submission rank sorted by human total</text>
      <text x="22" y="{height / 2:.2f}" transform="rotate(-90 22 {height / 2:.2f})" text-anchor="middle" fill="#374151" font-size="13">Mark</text>
    </svg>
  </div>
  <table>
    <thead>
      <tr><th>Rank</th><th>Participant</th><th>File</th><th>Human</th><th>Model</th></tr>
    </thead>
    <tbody>
      {''.join(labels_rows)}
    </tbody>
  </table>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    runs_path = Path(args.runs)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = load_rows(runs_path)
    output_path.write_text(build_html(rows), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
