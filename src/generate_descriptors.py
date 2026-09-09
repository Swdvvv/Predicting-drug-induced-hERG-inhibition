from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "data" / "processed" / "herg_model_dataset.csv"
DEFAULT_OUTPUT = ROOT_DIR / "data" / "processed" / "herg_descriptors.csv"
DEFAULT_REPORT = ROOT_DIR / "data" / "processed" / "herg_descriptor_report.csv"

DESCRIPTOR_COLUMNS = {
    "MolWt": Descriptors.MolWt,
    "LogP": Crippen.MolLogP,
    "HBD": Lipinski.NumHDonors,
    "HBA": Lipinski.NumHAcceptors,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "RotatableBonds": Lipinski.NumRotatableBonds,
    "RingCount": rdMolDescriptors.CalcNumRings,
    "AromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "HeavyAtomCount": Lipinski.HeavyAtomCount,
    "FormalCharge": Chem.GetFormalCharge,
    "NumAtoms": lambda molecule: molecule.GetNumAtoms(),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RDKit molecular descriptors for the hERG model dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the one-row-per-compound model CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the descriptor-augmented CSV.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Path for the descriptor-generation audit report.",
    )
    return parser.parse_args()


def load_dataset(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Model dataset not found: {input_path}")

    data = pd.read_csv(input_path)
    required_columns = {"molecule_chembl_id", "canonical_smiles"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return data


def calculate_descriptors(smiles: object) -> dict[str, float] | None:
    if pd.isna(smiles) or not str(smiles).strip():
        return None

    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return None

    return {
        name: float(function(molecule))
        for name, function in DESCRIPTOR_COLUMNS.items()
    }


def generate_descriptor_dataset(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    descriptor_rows: list[dict[str, float] | None] = []
    invalid_ids: list[str] = []

    for _, row in data.iterrows():
        descriptors = calculate_descriptors(row["canonical_smiles"])
        descriptor_rows.append(descriptors)
        if descriptors is None:
            invalid_ids.append(str(row["molecule_chembl_id"]))

    valid_mask = pd.Series(
        [descriptor_row is not None for descriptor_row in descriptor_rows],
        index=data.index,
    )
    valid_data = data.loc[valid_mask].copy()
    descriptor_data = pd.DataFrame(
        [descriptor_row for descriptor_row in descriptor_rows if descriptor_row is not None],
        index=valid_data.index,
    )
    descriptor_data.index = valid_data.index
    output = pd.concat([valid_data, descriptor_data], axis=1)

    report = pd.DataFrame(
        {
            "metric": [
                "input_records",
                "valid_smiles_records",
                "invalid_smiles_records",
                "output_records",
                "descriptor_count",
            ],
            "value": [
                len(data),
                len(output),
                len(invalid_ids),
                len(output),
                len(DESCRIPTOR_COLUMNS),
            ],
        }
    )
    if invalid_ids:
        report = pd.concat(
            [
                report,
                pd.DataFrame(
                    {
                        "metric": ["invalid_molecule_chembl_ids"],
                        "value": [", ".join(invalid_ids)],
                    }
                ),
            ],
            ignore_index=True,
        )
    return output.reset_index(drop=True), report


def main() -> None:
    arguments = parse_args()
    data = load_dataset(arguments.input)
    output, report = generate_descriptor_dataset(data)
    if output.empty:
        raise RuntimeError("No valid molecules remained after descriptor generation.")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(arguments.output, index=False)
    report.to_csv(arguments.report, index=False)

    print(f"Input records: {len(data):,}")
    print(f"Valid molecules: {len(output):,}")
    print(f"Invalid molecules: {len(data) - len(output):,}")
    print(f"Descriptors generated: {len(DESCRIPTOR_COLUMNS)}")
    print("\nDescriptor summary:")
    print(output[list(DESCRIPTOR_COLUMNS)].describe().round(3).to_string())
    print("\nMissing descriptor values:")
    print(output[list(DESCRIPTOR_COLUMNS)].isna().sum().to_string())
    print(f"\nDuplicate compounds: {output['molecule_chembl_id'].duplicated().sum():,}")
    print("\nMedian pIC50 summary:")
    print(output["median_pIC50"].describe().round(3).to_string())
    print(f"Descriptor dataset: {arguments.output}")
    print(f"Audit report: {arguments.report}")
    print("\nFirst five compounds:")
    print(
        output[["molecule_chembl_id", "median_pIC50", *DESCRIPTOR_COLUMNS]]
        .head()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
