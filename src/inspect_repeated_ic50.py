from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "data" / "processed" / "herg_chembl_curated.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "repeated_measurements"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect compounds with repeated exact IC50 measurements."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the curated IC50 CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for repeated-measurement tables and figures.",
    )
    return parser.parse_args()


def load_ic50_data(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Curated dataset not found: {input_path}")

    data = pd.read_csv(input_path)
    required_columns = {
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_value",
        "standard_units",
        "assay_chembl_id",
        "document_chembl_id",
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    data["standard_value"] = pd.to_numeric(data["standard_value"], errors="coerce")
    data = data[data["standard_value"] > 0].copy()
    if "pIC50" not in data:
        data["pIC50"] = 9 - np.log10(data["standard_value"])
    else:
        data["pIC50"] = pd.to_numeric(data["pIC50"], errors="coerce")
    return data


def build_repeated_measurement_table(data: pd.DataFrame) -> pd.DataFrame:
    counts = data.groupby("molecule_chembl_id").size().rename("measurement_count")
    repeated_ids = counts[counts > 1].index
    repeated = data[data["molecule_chembl_id"].isin(repeated_ids)].copy()

    summary = (
        repeated.groupby("molecule_chembl_id")
        .agg(
            number_of_measurements=("pIC50", "count"),
            mean_pIC50=("pIC50", "mean"),
            median_pIC50=("pIC50", "median"),
            min_pIC50=("pIC50", "min"),
            max_pIC50=("pIC50", "max"),
            std_pIC50=("pIC50", "std"),
            mean_IC50_nM=("standard_value", "mean"),
            min_IC50_nM=("standard_value", "min"),
            max_IC50_nM=("standard_value", "max"),
            unique_assays=("assay_chembl_id", "nunique"),
            unique_documents=("document_chembl_id", "nunique"),
            unique_SMILES=("canonical_smiles", "nunique"),
        )
        .reset_index()
    )
    summary["pIC50_range"] = summary["max_pIC50"] - summary["min_pIC50"]
    summary["variability"] = pd.cut(
        summary["pIC50_range"],
        bins=[-np.inf, 0.3, 1.0, np.inf],
        labels=["Low", "Moderate", "High"],
        right=False,
    ).astype(str)
    summary["multiple_assays"] = summary["unique_assays"] > 1
    summary["multiple_documents"] = summary["unique_documents"] > 1
    summary["multiple_SMILES"] = summary["unique_SMILES"] > 1
    return summary.sort_values(
        ["number_of_measurements", "pIC50_range"], ascending=[False, False]
    )


def save_outputs(
    data: pd.DataFrame,
    repeated_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    repeated_counts = repeated_summary[[
        "molecule_chembl_id",
        "number_of_measurements",
    ]]
    repeated_counts.to_csv(output_dir / "repeated_ic50_compounds.csv", index=False)

    repeated_ids = set(repeated_summary["molecule_chembl_id"])
    repeated_records = data[data["molecule_chembl_id"].isin(repeated_ids)].sort_values(
        ["molecule_chembl_id", "pIC50"], ascending=[True, False]
    )
    repeated_records.to_csv(output_dir / "repeated_ic50_measurements.csv", index=False)
    repeated_summary.to_csv(output_dir / "repeated_ic50_summary.csv", index=False)

    sns.set_theme(style="whitegrid", context="notebook")
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.histplot(
        data=repeated_summary,
        x="pIC50_range",
        bins=35,
        color="#c4512d",
        edgecolor="white",
        ax=axis,
    )
    axis.set_title("Spread of repeated IC50 measurements per compound")
    axis.set_xlabel("pIC50 range: maximum - minimum")
    axis.set_ylabel("Compounds")
    figure.tight_layout()
    figure.savefig(output_dir / "repeated_ic50_spread.png", dpi=200)
    plt.close(figure)

    top_compounds = repeated_summary.head(20).sort_values("number_of_measurements")
    figure, axis = plt.subplots(figsize=(10, 7))
    sns.barplot(
        data=top_compounds,
        x="number_of_measurements",
        y="molecule_chembl_id",
        color="#2f6f62",
        ax=axis,
    )
    axis.set_title("Compounds with the most repeated IC50 measurements")
    axis.set_xlabel("Exact IC50 measurements")
    axis.set_ylabel("ChEMBL compound ID")
    figure.tight_layout()
    figure.savefig(output_dir / "top_repeated_ic50_compounds.png", dpi=200)
    plt.close(figure)


def main() -> None:
    arguments = parse_args()
    data = load_ic50_data(arguments.input)
    repeated_summary = build_repeated_measurement_table(data)
    if repeated_summary.empty:
        raise RuntimeError("No compounds have repeated IC50 measurements.")

    save_outputs(data, repeated_summary, arguments.output_dir)

    print(f"IC50 records inspected: {len(data):,}")
    print(f"Compounds inspected: {data['molecule_chembl_id'].nunique():,}")
    print(f"Compounds with repeated measurements: {len(repeated_summary):,}")
    print(
        "IC50 measurements belonging to repeated compounds: "
        f"{repeated_summary['number_of_measurements'].sum():,}"
    )
    print("\nMeasurement variability:")
    print(repeated_summary["variability"].value_counts().to_string())
    print("\nAssay / document checks:")
    print(f"Compounds measured in multiple assays: {repeated_summary['multiple_assays'].sum()}")
    print(f"Compounds measured across multiple documents: {repeated_summary['multiple_documents'].sum()}")
    print(f"Compounds with multiple SMILES: {repeated_summary['multiple_SMILES'].sum()}")
    print(f"Output directory: {arguments.output_dir}")


if __name__ == "__main__":
    main()
