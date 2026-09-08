from pathlib import Path
from html import escape

import pandas as pd


INPUT_FILE = Path("data/raw/herg_chembl_clean.csv")
OUTPUT_FILE = Path("data/herg_dataset_report.html")


def make_bar_table(series, label):
    counts = series.fillna("Missing").astype(str).value_counts()
    maximum = counts.max()
    rows = []
    for name, count in counts.items():
        width = 0 if maximum == 0 else round(count / maximum * 100, 1)
        rows.append(
            f"<tr><td>{escape(str(name))}</td><td>{count:,}</td>"
            f"<td><span class='bar' style='width:{width}%'></span></td></tr>"
        )
    return (
        f"<table class='summary-table'><thead><tr><th>{label}</th>"
        "<th>Records</th><th>Relative volume</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    numeric_value = pd.to_numeric(df["standard_value"], errors="coerce")
    pchembl = pd.to_numeric(df["pchembl_value"], errors="coerce")

    overview = pd.DataFrame(
        [
            ["Activity records", f"{len(df):,}"],
            ["Unique compounds", f"{df['molecule_chembl_id'].nunique():,}"],
            ["Unique assays", f"{df['assay_chembl_id'].nunique():,}"],
            ["Unique documents", f"{df['document_chembl_id'].nunique():,}"],
            ["Activity value range", f"{numeric_value.min():,.2f} - {numeric_value.max():,.2f} nM"],
            ["Median activity value", f"{numeric_value.median():,.2f} nM"],
            ["pChEMBL values present", f"{pchembl.notna().sum():,}"],
        ],
        columns=["Metric", "Value"],
    )

    table_html = df.to_html(
        index=False,
        classes="dataset-table",
        table_id="dataset-table",
        na_rep="",
        escape=True,
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ChEMBL hERG Dataset Report</title>
<style>
:root {{
  --ink: #17211f;
  --muted: #64716d;
  --paper: #f4f1e9;
  --panel: #fffdf8;
  --line: #d9ded4;
  --accent: #c4512d;
  --accent-soft: #f2d4c5;
  --green: #2f6f62;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; color: var(--ink); background: var(--paper); font: 14px/1.45 Georgia, 'Times New Roman', serif; }}
main {{ max-width: 1500px; margin: 0 auto; padding: 38px 28px 60px; }}
header {{ border-bottom: 3px solid var(--ink); padding-bottom: 24px; margin-bottom: 24px; }}
h1 {{ margin: 0 0 8px; font: 700 clamp(30px, 4vw, 54px)/1.05 Georgia, serif; letter-spacing: 0; }}
.subtitle {{ color: var(--muted); font-size: 16px; margin: 0; }}
.grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 20px 0; }}
.card, section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 6px; }}
.card {{ padding: 16px; }}
.card strong {{ display: block; font: 700 25px/1.1 Georgia, serif; color: var(--accent); margin-top: 6px; }}
.card span {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
section {{ padding: 20px; margin-top: 18px; }}
h2 {{ margin: 0 0 14px; font-size: 23px; }}
.summary-grid {{ display: grid; grid-template-columns: minmax(280px, 1fr) minmax(280px, 1fr); gap: 24px; }}
.summary-table {{ border-collapse: collapse; width: 100%; }}
.summary-table th, .summary-table td {{ border-bottom: 1px solid var(--line); padding: 8px 6px; text-align: left; }}
.summary-table th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
.bar {{ display: block; height: 10px; min-width: 2px; background: var(--accent); border-radius: 2px; }}
.controls {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 14px; }}
input {{ width: min(520px, 100%); border: 1px solid var(--line); border-radius: 4px; padding: 10px 12px; background: white; color: var(--ink); font: inherit; }}
.note {{ color: var(--muted); font-size: 13px; }}
.table-wrap {{ overflow: auto; max-height: 72vh; border: 1px solid var(--line); }}
.dataset-table {{ border-collapse: collapse; min-width: 1300px; width: 100%; font-family: 'Segoe UI', sans-serif; font-size: 12px; }}
.dataset-table th {{ position: sticky; top: 0; z-index: 1; background: var(--green); color: white; text-align: left; white-space: nowrap; }}
.dataset-table th, .dataset-table td {{ padding: 8px 9px; border-bottom: 1px solid #e4e8e1; vertical-align: top; }}
.dataset-table tr:nth-child(even) {{ background: #f8faf5; }}
.dataset-table td:nth-child(2) {{ max-width: 280px; overflow-wrap: anywhere; }}
@media (max-width: 760px) {{ main {{ padding: 24px 14px 40px; }} .grid, .summary-grid {{ grid-template-columns: 1fr 1fr; }} .card strong {{ font-size: 20px; }} }}
@media (max-width: 480px) {{ .grid, .summary-grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<main>
<header>
  <h1>ChEMBL hERG activity dataset</h1>
  <p class="subtitle">Human KCNH2 / hERG activity records, filtered to compounds with structures and positive nM measurements.</p>
</header>
<div class="grid">
  <div class="card"><span>Activity records</span><strong>{len(df):,}</strong></div>
  <div class="card"><span>Unique compounds</span><strong>{df['molecule_chembl_id'].nunique():,}</strong></div>
  <div class="card"><span>Unique assays</span><strong>{df['assay_chembl_id'].nunique():,}</strong></div>
  <div class="card"><span>pChEMBL coverage</span><strong>{pchembl.notna().mean() * 100:.1f}%</strong></div>
</div>
<section>
  <h2>Dataset overview</h2>
  {overview.to_html(index=False, classes='summary-table', escape=True)}
</section>
<section class="summary-grid">
  <div><h2>Activity types</h2>{make_bar_table(df['standard_type'], 'Type')}</div>
  <div><h2>Assay types</h2>{make_bar_table(df['assay_type'], 'Assay')}</div>
</section>
<section>
  <h2>All records</h2>
  <div class="controls">
    <input id="filter" type="search" placeholder="Filter rows by compound, SMILES, assay, activity type...">
    <span id="count" class="note"></span>
  </div>
  <div class="table-wrap">{table_html}</div>
</section>
</main>
<script>
const input = document.getElementById('filter');
const rows = Array.from(document.querySelectorAll('#dataset-table tbody tr'));
const count = document.getElementById('count');
function updateRows() {{
  const query = input.value.trim().toLowerCase();
  let visible = 0;
  rows.forEach(row => {{
    const match = !query || row.textContent.toLowerCase().includes(query);
    row.hidden = !match;
    if (match) visible += 1;
  }});
  count.textContent = `${{visible.toLocaleString()}} of ${{rows.length.toLocaleString()}} records shown`;
}}
input.addEventListener('input', updateRows);
updateRows();
</script>
</body>
</html>
"""

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(html, encoding="utf-8")
    print(f"Created {OUTPUT_FILE} ({OUTPUT_FILE.stat().st_size / 1_048_576:.2f} MB)")


if __name__ == "__main__":
    main()
