from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "data" / "raw" / "herg_chembl_clean.csv"
DEFAULT_OUTPUT = ROOT_DIR / "results" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create visualisations for the cleaned ChEMBL hERG dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the cleaned hERG CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory where figures and the summary CSV are written.",
    )
    return parser.parse_args()


def load_dataset(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    data = pd.read_csv(input_path)
    required_columns = {
        "molecule_chembl_id",
        "standard_type",
        "standard_value",
        "standard_units",
        "pchembl_value",
        "assay_type",
        "assay_chembl_id",
        "document_chembl_id",
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    data["standard_value"] = pd.to_numeric(data["standard_value"], errors="coerce")
    data["pchembl_value"] = pd.to_numeric(data["pchembl_value"], errors="coerce")
    data = data[data["standard_value"] > 0].copy()
    data["log10_standard_value"] = data["standard_value"].apply(math.log10)
    return data


def save_summary(data: pd.DataFrame, output_dir: Path) -> None:
    summary = pd.DataFrame(
        {
            "metric": [
                "activity_records",
                "unique_compounds",
                "unique_assays",
                "unique_documents",
                "median_standard_value_nm",
                "median_pchembl_value",
            ],
            "value": [
                len(data),
                data["molecule_chembl_id"].nunique(),
                data["assay_chembl_id"].nunique(),
                data["document_chembl_id"].nunique(),
                data["standard_value"].median(),
                data["pchembl_value"].median(),
            ],
        }
    )
    summary.to_csv(output_dir / "herg_summary.csv", index=False)


def create_figures(data: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    output_dir.mkdir(parents=True, exist_ok=True)

    activity_data = data.dropna(subset=["log10_standard_value"])
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.histplot(
        data=activity_data,
        x="log10_standard_value",
        bins=45,
        color="#c4512d",
        edgecolor="white",
        ax=axis,
    )
    axis.set_title("Distribution of hERG activity values")
    axis.set_xlabel("log10 standard value (nM)")
    axis.set_ylabel("Activity records")
    figure.tight_layout()
    figure.savefig(output_dir / "activity_distribution.png", dpi=180)
    plt.close(figure)

    type_counts = data["standard_type"].fillna("Missing").value_counts().head(15).sort_values()
    figure, axis = plt.subplots(figsize=(10, 6))
    type_counts.plot.barh(ax=axis, color="#2f6f62")
    axis.set_title("Most common activity types")
    axis.set_xlabel("Activity records")
    axis.set_ylabel("Standard type")
    figure.tight_layout()
    figure.savefig(output_dir / "activity_types.png", dpi=180)
    plt.close(figure)

    pchembl_data = data.dropna(subset=["pchembl_value"])
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.histplot(
        data=pchembl_data,
        x="pchembl_value",
        bins=35,
        color="#2f6f62",
        edgecolor="white",
        ax=axis,
    )
    axis.set_title("Distribution of pChEMBL values")
    axis.set_xlabel("pChEMBL value")
    axis.set_ylabel("Activity records")
    figure.tight_layout()
    figure.savefig(output_dir / "pchembl_distribution.png", dpi=180)
    plt.close(figure)

    assay_counts = data["assay_chembl_id"].value_counts().head(20).sort_values()
    figure, axis = plt.subplots(figsize=(10, 8))
    assay_counts.plot.barh(ax=axis, color="#7d8f4e")
    axis.set_title("Top 20 assays by number of measurements")
    axis.set_xlabel("Activity records")
    axis.set_ylabel("Assay ChEMBL ID")
    figure.tight_layout()
    figure.savefig(output_dir / "records_by_assay.png", dpi=180)
    plt.close(figure)

    assay_type_counts = data["assay_type"].fillna("Missing").value_counts()
    figure, axis = plt.subplots(figsize=(8, 5))
    assay_type_counts.sort_values().plot.barh(ax=axis, color="#c4512d")
    axis.set_title("hERG assay types")
    axis.set_xlabel("Activity records")
    axis.set_ylabel("Assay type")
    figure.tight_layout()
    figure.savefig(output_dir / "assay_types.png", dpi=180)
    plt.close(figure)

    compound_counts = data["molecule_chembl_id"].value_counts().head(20).sort_values()
    figure, axis = plt.subplots(figsize=(10, 6))
    compound_counts.plot.barh(ax=axis, color="#2f6f62")
    axis.set_title("Top 20 compounds by number of measurements")
    axis.set_xlabel("Activity records")
    axis.set_ylabel("ChEMBL compound ID")
    figure.tight_layout()
    figure.savefig(output_dir / "top_compounds_measurements.png", dpi=180)
    plt.close(figure)

    activity_by_type = data[data["standard_type"].isin(["IC50", "Ki"])].copy()
    activity_by_type = activity_by_type.dropna(subset=["standard_value"])
    figure, axis = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        data=activity_by_type,
        x="standard_type",
        y="standard_value",
        color="#d98d6d",
        ax=axis,
    )
    axis.set_yscale("log")
    axis.set_title("hERG activity by measurement type")
    axis.set_xlabel("Activity type")
    axis.set_ylabel("Activity (nM, log scale)")
    figure.tight_layout()
    figure.savefig(output_dir / "activity_by_type.png", dpi=180)
    plt.close(figure)


def main() -> None:
    arguments = parse_args()
    data = load_dataset(arguments.input)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    save_summary(data, arguments.output_dir)
    create_figures(data, arguments.output_dir)
    print(f"Created visualisations in {arguments.output_dir}")
    print(f"Records visualised: {len(data):,}")
    print("Generated files:")
    for file in sorted(arguments.output_dir.glob("*.png")):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
