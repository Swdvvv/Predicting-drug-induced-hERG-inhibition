from pathlib import Path
import time

import pandas as pd
import requests
from tqdm import tqdm


BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
UNIPROT_ACCESSION = "Q12809"
OUTPUT_DIR = Path("data/raw")
OUTPUT_FILE = OUTPUT_DIR / "herg_chembl_clean.csv"
PAGE_SIZE = 1000
MOLECULE_BATCH_SIZE = 50
REQUEST_DELAY = 0.1


def get_json(url, params=None, retries=3):
    """Send a GET request to the ChEMBL API."""
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            if attempt == retries - 1:
                raise RuntimeError(f"ChEMBL request failed: {error}") from error
            time.sleep(2 ** attempt)


def find_herg_target():
    """Identify the human KCNH2 target using its UniProt accession."""
    print("\nSearching ChEMBL for human KCNH2/hERG...")
    data = get_json(
        f"{BASE_URL}/target.json",
        {"target_components__accession": UNIPROT_ACCESSION, "limit": 100},
    )
    targets = data.get("targets", [])
    if not targets:
        raise RuntimeError("No ChEMBL target found for UniProt Q12809.")

    print(f"Found {len(targets)} candidate targets:\n")
    candidates = []
    for target in targets:
        target_id = target.get("target_chembl_id")
        name = target.get("pref_name")
        organism = target.get("organism")
        target_type = target.get("target_type")
        print(f"{target_id} | {name} | {organism} | {target_type}")
        if organism == "Homo sapiens" and target_type == "SINGLE PROTEIN":
            candidates.append(target)

    if not candidates:
        raise RuntimeError("Could not identify a human SINGLE PROTEIN KCNH2 target.")

    def target_score(target):
        name = target.get("pref_name", "").lower()
        return ("herg" in name, "kcnh2" in name, "potassium" in name)

    candidates.sort(key=target_score, reverse=True)
    selected = candidates[0]
    print("\nSelected target:")
    print(f"ChEMBL ID : {selected['target_chembl_id']}")
    print(f"Name      : {selected['pref_name']}")
    print(f"Organism  : {selected['organism']}")
    print(f"Type      : {selected['target_type']}")
    return selected["target_chembl_id"]


def download_activities(target_id):
    """Download all activity records associated with the selected target."""
    print("\nDownloading activity records...")
    records = []
    offset = 0
    while True:
        data = get_json(
            f"{BASE_URL}/activity.json",
            {"target_chembl_id": target_id, "limit": PAGE_SIZE, "offset": offset},
        )
        page = data.get("activities", [])
        if not page:
            break
        records.extend(page)
        print(f"Downloaded {len(records):,} records...")
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(REQUEST_DELAY)
    print(f"\nTotal activity records: {len(records):,}")
    return records


def download_smiles(molecule_ids):
    """Download canonical SMILES for all molecules."""
    print(f"\nDownloading structures for {len(molecule_ids):,} molecules...")
    smiles = {}
    molecule_ids = list(molecule_ids)
    for start in tqdm(range(0, len(molecule_ids), MOLECULE_BATCH_SIZE), desc="Molecules"):
        batch = molecule_ids[start:start + MOLECULE_BATCH_SIZE]
        data = get_json(
            f"{BASE_URL}/molecule.json",
            {"molecule_chembl_id__in": ",".join(batch), "limit": MOLECULE_BATCH_SIZE},
        )
        for molecule in data.get("molecules", []):
            molecule_id = molecule.get("molecule_chembl_id")
            structures = molecule.get("molecule_structures")
            if structures:
                canonical_smiles = structures.get("canonical_smiles")
                if canonical_smiles:
                    smiles[molecule_id] = canonical_smiles
        time.sleep(REQUEST_DELAY)
    print(f"Retrieved structures for {len(smiles):,} molecules.")
    return smiles


def clean_data(records, smiles):
    """Create a clean activity dataset with valid structures and values."""
    print("\nCleaning data...")
    rows = []
    for record in records:
        molecule_id = record.get("molecule_chembl_id")
        standard_value = record.get("standard_value")
        if molecule_id not in smiles or standard_value is None:
            continue
        try:
            standard_value = float(standard_value)
        except (ValueError, TypeError):
            continue
        if standard_value <= 0:
            continue
        rows.append({
            "molecule_chembl_id": molecule_id,
            "canonical_smiles": smiles[molecule_id],
            "standard_type": record.get("standard_type"),
            "standard_relation": record.get("standard_relation"),
            "standard_value": standard_value,
            "standard_units": record.get("standard_units"),
            "pchembl_value": record.get("pchembl_value"),
            "assay_chembl_id": record.get("assay_chembl_id"),
            "document_chembl_id": record.get("document_chembl_id"),
            "target_chembl_id": record.get("target_chembl_id"),
            "target_pref_name": record.get("target_pref_name"),
            "target_organism": record.get("target_organism"),
            "assay_type": record.get("assay_type"),
            "activity_comment": record.get("activity_comment"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No usable activity records were found.")
    df["standard_units"] = df["standard_units"].astype(str).str.strip().str.lower()
    df = df[df["standard_units"].isin(["nm", "nanomolar"])].copy()
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df):,} exact duplicates.")
    return df.sort_values(
        by=["molecule_chembl_id", "standard_type", "standard_value"]
    ).reset_index(drop=True)


def main():
    print("=" * 60)
    print("ChEMBL HUMAN hERG/KCNH2 DATASET")
    print("=" * 60)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_id = find_herg_target()
    records = download_activities(target_id)
    molecule_ids = {
        record.get("molecule_chembl_id")
        for record in records
        if record.get("molecule_chembl_id")
    }
    print(f"\nUnique compounds identified: {len(molecule_ids):,}")
    smiles = download_smiles(molecule_ids)
    df = clean_data(records, smiles)
    df.to_csv(OUTPUT_FILE, index=False)
    print("\n" + "=" * 60)
    print("DATASET COMPLETE")
    print("=" * 60)
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Activity records: {len(df):,}")
    print("Unique compounds:", df["molecule_chembl_id"].nunique())
    print("\nActivity types:")
    print(df["standard_type"].value_counts().to_string())
    print("\nActivity units:")
    print(df["standard_units"].value_counts().to_string())
    print("\nFirst five rows:")
    print(df.head().to_string())
    print("\nFinished successfully.")


if __name__ == "__main__":
    main()
