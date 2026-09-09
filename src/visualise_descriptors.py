from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "data" / "processed" / "herg_descriptors.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "figures" / "descriptors"

DESCRIPTOR_COLUMNS = [
    "MolWt",
    "LogP",
    "HBD",
    "HBA",
    "TPSA",
    "RotatableBonds",
    "RingCount",
    "AromaticRings",
    "FractionCSP3",
    "HeavyAtomCount",
    "FormalCharge",
    "NumAtoms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise RDKit descriptors and their relationship with median pIC50."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the descriptor CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for descriptor figures and summary files.",
    )
    return parser.parse_args()


def load_data(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Descriptor dataset not found: {input_path}")

    data = pd.read_csv(input_path)
    required_columns = {"molecule_chembl_id", "median_pIC50", *DESCRIPTOR_COLUMNS}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    numeric_columns = ["median_pIC50", *DESCRIPTOR_COLUMNS]
    data[numeric_columns] = data[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    return data.dropna(subset=numeric_columns).copy()


def save_summary(data: pd.DataFrame, output_dir: Path) -> None:
    summary = data[["median_pIC50", *DESCRIPTOR_COLUMNS]].describe().T
    summary.to_csv(output_dir / "descriptor_summary.csv")

    correlations = (
        data[DESCRIPTOR_COLUMNS + ["median_pIC50"]]
        .corr()["median_pIC50"]
        .drop("median_pIC50")
        .sort_values(key=abs, ascending=False)
    )
    correlations.rename("correlation").to_csv(
        output_dir / "descriptor_pIC50_correlations.csv",
        header=True,
    )

    spearman = data[["median_pIC50", *DESCRIPTOR_COLUMNS]].corr(method="spearman")
    spearman.to_csv(output_dir / "descriptor_spearman_correlations.csv")


def create_figures(data: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")

    for descriptor in DESCRIPTOR_COLUMNS:
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.hist(data[descriptor].dropna(), bins=40, color="#2f6f62")
        axis.set_xlabel(descriptor)
        axis.set_ylabel("Number of compounds")
        axis.set_title(f"Distribution of {descriptor}")
        figure.tight_layout()
        figure.savefig(output_dir / f"{descriptor}_distribution.png", dpi=300)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(8, 5))
        axis.scatter(
            data[descriptor],
            data["median_pIC50"],
            alpha=0.4,
            s=12,
            color="#c4512d",
            edgecolors="none",
        )
        axis.set_xlabel(descriptor)
        axis.set_ylabel("Median pIC50")
        axis.set_title(f"{descriptor} vs hERG pIC50")
        figure.tight_layout()
        figure.savefig(output_dir / f"{descriptor}_vs_pIC50.png", dpi=300)
        plt.close(figure)

    figure, axes = plt.subplots(3, 4, figsize=(16, 11))
    for axis, descriptor in zip(axes.flat, DESCRIPTOR_COLUMNS):
        sns.histplot(data=data, x=descriptor, bins=30, color="#2f6f62", ax=axis)
        axis.set_title(descriptor)
        axis.set_xlabel("")
        axis.set_ylabel("Compounds")
    figure.suptitle("RDKit descriptor distributions", y=1.01, fontsize=16)
    figure.tight_layout()
    figure.savefig(output_dir / "descriptor_distributions.png", dpi=200)
    plt.close(figure)

    correlations = data[["median_pIC50", *DESCRIPTOR_COLUMNS]].corr(method="spearman")
    figure, axis = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        correlations,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.3,
        ax=axis,
    )
    axis.set_title("Spearman correlations: descriptors and median pIC50")
    figure.tight_layout()
    figure.savefig(output_dir / "descriptor_correlation_heatmap.png", dpi=200)
    plt.close(figure)

    selected = ["MolWt", "LogP", "TPSA", "HBA", "FractionCSP3", "NumAtoms"]
    figure, axes = plt.subplots(2, 3, figsize=(16, 10))
    for axis, descriptor in zip(axes.flat, selected):
        sns.scatterplot(
            data=data,
            x=descriptor,
            y="median_pIC50",
            alpha=0.35,
            s=20,
            color="#c4512d",
            edgecolor=None,
            ax=axis,
        )
        axis.set_title(f"median pIC50 vs {descriptor}")
        axis.set_ylabel("Median pIC50")
    figure.suptitle("Activity relationships with selected descriptors", y=1.01, fontsize=16)
    figure.tight_layout()
    figure.savefig(output_dir / "pIC50_vs_descriptors.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 6))
    sns.histplot(data=data, x="median_pIC50", bins=35, color="#c4512d", ax=axis)
    axis.set_title("Distribution of compound-level median pIC50")
    axis.set_xlabel("Median pIC50")
    axis.set_ylabel("Compounds")
    figure.tight_layout()
    figure.savefig(output_dir / "median_pIC50_distribution.png", dpi=200)
    plt.close(figure)


def main() -> None:
    arguments = parse_args()
    data = load_data(arguments.input)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    save_summary(data, arguments.output_dir)
    create_figures(data, arguments.output_dir)

    print(f"Compounds visualised: {len(data):,}")
    print(f"Descriptors visualised: {len(DESCRIPTOR_COLUMNS)}")
    print(f"Output directory: {arguments.output_dir}")
    print("Generated files:")
    for file in sorted(arguments.output_dir.iterdir()):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
