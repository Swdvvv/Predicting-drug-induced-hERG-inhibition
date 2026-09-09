from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "data" / "processed" / "herg_chembl_curated.csv"
DEFAULT_OUTPUT = ROOT_DIR / "data" / "processed" / "herg_model_dataset.csv"
DEFAULT_REPORT = ROOT_DIR / "data" / "processed" / "herg_model_dataset_report.csv"

REQUIRED_COLUMNS = {
    "molecule_chembl_id",
    "canonical_smiles",
    "standard_value",
    "pIC50",
    "assay_chembl_id",
    "document_chembl_id",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a one-row-per-compound hERG ML dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the curated IC50 dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the one-row-per-compound model dataset.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Path for the model-dataset audit report.",
    )
    parser.add_argument(
        "--aggregation",
        choices=["median", "mean"],
        default="median",
        help="Retained for compatibility; the output includes both mean and median pIC50.",
    )
    return parser.parse_args()


def load_data(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Curated dataset not found: {input_path}")

    data = pd.read_csv(input_path)
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    data["standard_value"] = pd.to_numeric(data["standard_value"], errors="coerce")
    data["pIC50"] = pd.to_numeric(data["pIC50"], errors="coerce")
    data = data.dropna(
        subset=["molecule_chembl_id", "canonical_smiles", "standard_value", "pIC50"]
    ).copy()
    data = data[data["standard_value"] > 0]
    data = data[data["canonical_smiles"].astype(str).str.strip().ne("")]
    return data


def create_model_dataset(
    data: pd.DataFrame,
    aggregation: str = "median",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = data.groupby("molecule_chembl_id", sort=False)
    smiles_counts = grouped["canonical_smiles"].nunique()
    conflicting_smiles = smiles_counts[smiles_counts > 1]

    summary = grouped.agg(
        number_of_measurements=("pIC50", "count"),
        median_pIC50=("pIC50", "median"),
        mean_pIC50=("pIC50", "mean"),
        min_pIC50=("pIC50", "min"),
        max_pIC50=("pIC50", "max"),
        pIC50_std=("pIC50", "std"),
        unique_assays=("assay_chembl_id", "nunique"),
        unique_documents=("document_chembl_id", "nunique"),
        min_IC50_nM=("standard_value", "min"),
        max_IC50_nM=("standard_value", "max"),
        canonical_smiles=("canonical_smiles", "first"),
    )
    summary["pIC50_range"] = summary["max_pIC50"] - summary["min_pIC50"]
    summary["pIC50_std"] = summary["pIC50_std"].fillna(0)
    summary["variability"] = pd.cut(
        summary["pIC50_range"],
        bins=[-float("inf"), 0.3, 1.0, float("inf")],
        labels=["Low", "Moderate", "High"],
        right=False,
    ).astype(str)
    summary = summary.reset_index()

    model_columns = [
        "molecule_chembl_id",
        "canonical_smiles",
        "median_pIC50",
        "mean_pIC50",
        "min_pIC50",
        "max_pIC50",
        "number_of_measurements",
        "pIC50_std",
        "pIC50_range",
        "variability",
        "unique_assays",
        "unique_documents",
        "min_IC50_nM",
        "max_IC50_nM",
    ]
    model_data = summary[model_columns].sort_values(
        "molecule_chembl_id"
    ).reset_index(drop=True)

    report = pd.DataFrame(
        {
            "metric": [
                "input_records",
                "input_unique_compounds",
                "output_records",
                "repeated_compounds_aggregated",
                "conflicting_smiles_compounds",
                "multiple_smiles_compounds",
                "variability_low",
                "variability_moderate",
                "variability_high",
            ],
            "value": [
                len(data),
                data["molecule_chembl_id"].nunique(),
                len(model_data),
                int((model_data["number_of_measurements"] > 1).sum()),
                len(conflicting_smiles),
                len(conflicting_smiles),
                int((model_data["variability"] == "Low").sum()),
                int((model_data["variability"] == "Moderate").sum()),
                int((model_data["variability"] == "High").sum()),
            ],
        }
    )
    return model_data, report


def main() -> None:
    arguments = parse_args()
    data = load_data(arguments.input)
    model_data, report = create_model_dataset(data, arguments.aggregation)
    if model_data.empty:
        raise RuntimeError("No model records were created.")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    model_data.to_csv(arguments.output, index=False)
    report.to_csv(arguments.report, index=False)

    print(f"Input records: {len(data):,}")
    print(f"Input unique compounds: {data['molecule_chembl_id'].nunique():,}")
    print(f"Model records: {len(model_data):,}")
    print(
        "Repeated compounds aggregated: "
        f"{(model_data['number_of_measurements'] > 1).sum():,}"
    )
    multiple_smiles = (
        data.groupby("molecule_chembl_id")["canonical_smiles"].nunique() > 1
    ).sum()
    print(f"Compounds with multiple SMILES: {multiple_smiles}")
    print("Variability categories:")
    print(model_data["variability"].value_counts().to_string())
    print(f"Model dataset: {arguments.output}")
    print(f"Audit report: {arguments.report}")


if __name__ == "__main__":
    main()
