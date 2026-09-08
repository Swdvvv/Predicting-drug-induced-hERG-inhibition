from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "data" / "raw" / "herg_chembl_clean.csv"
DEFAULT_OUTPUT = ROOT_DIR / "data" / "processed" / "herg_chembl_curated.csv"
DEFAULT_REPORT = ROOT_DIR / "data" / "processed" / "herg_curation_report.csv"

REQUIRED_COLUMNS = {
    "molecule_chembl_id",
    "canonical_smiles",
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Curate the cleaned ChEMBL hERG activity dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the input hERG CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the curated CSV file.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Path for the curation report CSV file.",
    )
    parser.add_argument(
        "--keep-all-types",
        action="store_true",
        help="Keep all activity types instead of the default IC50-only dataset.",
    )
    return parser.parse_args()


def load_data(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    data = pd.read_csv(input_path)
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input dataset is missing required columns: {missing}")
    return data


def curate_data(
    data: pd.DataFrame,
    keep_all_types: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = data.copy()
    report: list[dict[str, int | str]] = []

    def record(step: str, before: int, after: int) -> None:
        report.append(
            {
                "step": step,
                "records_before": before,
                "records_after": after,
                "records_removed": before - after,
            }
        )

    before = len(working)
    working = working.dropna(subset=["molecule_chembl_id", "canonical_smiles"])
    working = working[working["canonical_smiles"].astype(str).str.strip().ne("")]
    record("valid compound identifiers and structures", before, len(working))

    before = len(working)
    working["standard_value"] = pd.to_numeric(
        working["standard_value"], errors="coerce"
    )
    working = working.dropna(subset=["standard_value"])
    working = working[working["standard_value"] > 0]
    record("positive numeric activity values", before, len(working))

    before = len(working)
    units = working["standard_units"].astype(str).str.strip().str.lower()
    working = working[units.isin({"nm", "nanomolar"})].copy()
    working["standard_units"] = "nM"
    record("nanomolar activity units", before, len(working))

    if not keep_all_types:
        before = len(working)
        working = working[
            working["standard_type"].fillna("").astype(str).str.upper().eq("IC50")
        ].copy()
        record("IC50 activity type", before, len(working))

    before = len(working)
    working = working[
        working["standard_relation"].fillna("").astype(str).str.strip().eq("=")
    ].copy()
    record("exact standard relations", before, len(working))

    before = len(working)
    working = working.drop_duplicates().copy()
    record("exact duplicate rows removed", before, len(working))

    working["standard_type"] = working["standard_type"].astype(str).str.strip()
    working["standard_relation"] = working["standard_relation"].fillna("").astype(str).str.strip()
    if not keep_all_types:
        working["pIC50"] = 9 - np.log10(working["standard_value"])

    columns_to_keep = [
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
    ]
    if not keep_all_types:
        columns_to_keep.append("pIC50")
    columns_to_keep.extend(
        [
            "pchembl_value",
            "assay_chembl_id",
            "document_chembl_id",
            "target_chembl_id",
            "target_pref_name",
            "target_organism",
            "assay_type",
            "activity_comment",
        ]
    )
    working = working[columns_to_keep]
    working = working.sort_values(
        ["molecule_chembl_id", "standard_value"],
        ascending=[True, False],
    ).reset_index(drop=True)

    report_data = pd.DataFrame(report)
    return working, report_data


def main() -> None:
    arguments = parse_args()
    data = load_data(arguments.input)
    curated, report = curate_data(
        data,
        keep_all_types=arguments.keep_all_types,
    )
    if curated.empty:
        raise RuntimeError("Curation removed every record; no output was written.")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    curated.to_csv(arguments.output, index=False)
    report.to_csv(arguments.report, index=False)

    print(f"Input records: {len(data):,}")
    print(f"Curated records: {len(curated):,}")
    print(f"Unique compounds: {curated['molecule_chembl_id'].nunique():,}")
    if "pIC50" in curated:
        print(f"pIC50 range: {curated['pIC50'].min():.2f} - {curated['pIC50'].max():.2f}")
    print(f"Curated dataset: {arguments.output}")
    print(f"Curation report: {arguments.report}")


if __name__ == "__main__":
    main()
