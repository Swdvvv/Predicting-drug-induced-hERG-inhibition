from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "data" / "processed" / "herg_model_dataset.csv"
DEFAULT_OUTPUT = ROOT_DIR / "data" / "processed" / "herg_fingerprints.csv"
DEFAULT_REPORT = ROOT_DIR / "data" / "processed" / "herg_fingerprint_report.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Morgan fingerprints for the hERG model compounds."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the descriptor or model CSV containing canonical SMILES.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the fingerprint bit-matrix CSV.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Path for the fingerprint-generation audit report.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius. Default: 2.",
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        default=2048,
        help="Fingerprint length in bits. Default: 2048.",
    )
    return parser.parse_args()


def load_data(input_path: Path) -> tuple[pd.DataFrame, str]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    data = pd.read_csv(input_path)
    id_column = next(
        (
            column
            for column in ("molregno", "molecule_chembl_id")
            if column in data.columns
        ),
        None,
    )
    required_columns = {"canonical_smiles", "median_pIC50"}
    if id_column is None:
        required_columns.add("molregno")
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if data[id_column].duplicated().any():
        duplicate_count = int(data[id_column].duplicated().sum())
        raise ValueError(f"Found {duplicate_count:,} duplicate compound IDs.")
    return data, id_column


def generate_fingerprints(
    data: pd.DataFrame,
    id_column: str,
    radius: int,
    n_bits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if radius < 0:
        raise ValueError("Fingerprint radius must be non-negative.")
    if n_bits <= 0:
        raise ValueError("Fingerprint length must be positive.")

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
    )
    fingerprint_rows: list[list[int]] = []
    invalid_smiles = 0

    for _, row in data.iterrows():
        smiles = row["canonical_smiles"]
        molecule = None if pd.isna(smiles) else Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            invalid_smiles += 1
            fingerprint_rows.append([0] * n_bits)
            continue

        fingerprint = generator.GetFingerprint(molecule)
        fingerprint_rows.append(list(fingerprint))

    if invalid_smiles:
        raise ValueError(
            f"Invalid SMILES were detected: {invalid_smiles:,}. "
            "Fingerprint generation stopped."
        )

    bit_columns = [f"Morgan_{bit}" for bit in range(n_bits)]
    fingerprint_data = pd.DataFrame(fingerprint_rows, columns=bit_columns)
    output = pd.concat(
        [data[[id_column, "canonical_smiles", "median_pIC50"]].reset_index(drop=True), fingerprint_data],
        axis=1,
    )
    duplicate_fingerprints = int(fingerprint_data.duplicated().sum())
    active_bits = int(fingerprint_data.to_numpy().sum())
    total_bits = n_bits * len(output)

    report = pd.DataFrame(
        {
            "metric": [
                "input_records",
                "output_records",
                "fingerprint_radius",
                "fingerprint_bits",
                "invalid_smiles",
                "duplicate_compound_ids",
                "duplicate_fingerprints",
                "total_fingerprint_bits",
                "active_fingerprint_bits",
                "mean_active_bits_per_compound",
            ],
            "value": [
                len(data),
                len(output),
                radius,
                n_bits,
                invalid_smiles,
                int(data[id_column].duplicated().sum()),
                duplicate_fingerprints,
                total_bits,
                active_bits,
                active_bits / len(output),
            ],
        }
    )
    return output, report


def main() -> None:
    arguments = parse_args()
    data, id_column = load_data(arguments.input)
    output, report = generate_fingerprints(
        data,
        id_column=id_column,
        radius=arguments.radius,
        n_bits=arguments.n_bits,
    )
    if output.empty:
        raise RuntimeError("No valid molecules remained after fingerprint generation.")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(arguments.output, index=False)
    report.to_csv(arguments.report, index=False)

    print(f"Input records: {len(data):,}")
    print(f"Valid molecules: {len(output):,}")
    print(f"Fingerprint type: Morgan radius {arguments.radius}")
    print(f"Fingerprint bits: {arguments.n_bits}")
    print("Invalid SMILES: 0")
    print(f"Duplicate compounds: {int(data[id_column].duplicated().sum()):,}")
    print(f"Duplicate fingerprints: {int(output.iloc[:, 3:].duplicated().sum()):,}")
    print(f"Fingerprint dataset: {arguments.output}")
    print(f"Audit report: {arguments.report}")


if __name__ == "__main__":
    main()
