from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT_DIR = Path(__file__).resolve().parents[1]

SPLIT_DIR = ROOT_DIR / "data" / "processed" / "splits"
OUTPUT_DIR = ROOT_DIR / "results"

DESCRIPTOR_PATH = ROOT_DIR / "data" / "processed" / "herg_descriptors.csv"
FINGERPRINT_PATH = ROOT_DIR / "data" / "processed" / "herg_fingerprints.csv"

RANDOM_STATE = 42
N_ESTIMATORS = 500

ID_COLUMN = "molecule_chembl_id"

DESCRIPTORS = [
    "MolWt",
    "LogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotatableBonds",
    "RingCount",
    "AromaticRings",
    "FractionCSP3",
    "HeavyAtomCount",
    "FormalCharge",
    "NumAtoms",
]

MORGAN_FEATURES = [f"Morgan_{i}" for i in range(2048)]
FEATURES = DESCRIPTORS + MORGAN_FEATURES


def calculate_metrics(y_true, predictions):
    return {
        "n_test": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
        "pearson_r": float(np.corrcoef(y_true, predictions)[0, 1]),
    }


def main():
    print("=" * 60)
    print("hERG COMBINED DESCRIPTORS + MORGAN RANDOM FOREST")
    print("=" * 60)

    print("\nLoading descriptor dataset...")
    descriptors = pd.read_csv(DESCRIPTOR_PATH)
    print(f"Descriptor compounds: {len(descriptors):,}")

    print("\nLoading Morgan fingerprint dataset...")
    fingerprints = pd.read_csv(FINGERPRINT_PATH)
    print(f"Fingerprint compounds: {len(fingerprints):,}")

    if ID_COLUMN not in descriptors.columns:
        raise ValueError(f"{ID_COLUMN} not found in descriptor dataset.")
    if ID_COLUMN not in fingerprints.columns:
        raise ValueError(f"{ID_COLUMN} not found in fingerprint dataset.")

    missing_descriptors = [
        column for column in DESCRIPTORS if column not in descriptors.columns
    ]
    missing_morgan = [
        column for column in MORGAN_FEATURES if column not in fingerprints.columns
    ]
    if missing_descriptors:
        raise ValueError(f"Missing descriptor columns: {missing_descriptors}")
    if missing_morgan:
        raise ValueError(f"Missing Morgan fingerprint columns: {missing_morgan[:10]}")

    print("\nMerging descriptors and Morgan fingerprints...")
    feature_df = descriptors[
        [ID_COLUMN, "canonical_smiles", "median_pIC50"] + DESCRIPTORS
    ].merge(
        fingerprints[[ID_COLUMN] + MORGAN_FEATURES],
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )
    print(f"Merged compounds: {len(feature_df):,}")

    if len(feature_df) != len(descriptors):
        raise ValueError("Merged dataset size does not match descriptor dataset.")
    if len(feature_df) != len(fingerprints):
        raise ValueError("Merged dataset size does not match fingerprint dataset.")

    print("\nLoading the EXACT same random split...")
    train_split = pd.read_csv(SPLIT_DIR / "random_train.csv")
    test_split = pd.read_csv(SPLIT_DIR / "random_test.csv")
    train_ids = set(train_split[ID_COLUMN])
    test_ids = set(test_split[ID_COLUMN])

    if train_ids & test_ids:
        raise ValueError("Training and test sets contain overlapping compounds.")

    train = feature_df[feature_df[ID_COLUMN].isin(train_ids)].copy()
    test = feature_df[feature_df[ID_COLUMN].isin(test_ids)].copy()
    print(f"Training compounds: {len(train):,}")
    print(f"Test compounds:     {len(test):,}")

    if len(train) != len(train_split):
        raise ValueError("Training split size does not match expected split.")
    if len(test) != len(test_split):
        raise ValueError("Test split size does not match expected split.")

    print("\nPreparing combined feature matrix...")
    print(f"Descriptors:        {len(DESCRIPTORS)}")
    print(f"Morgan bits:        {len(MORGAN_FEATURES)}")
    print(f"Total features:     {len(FEATURES)}")

    x_train = train[FEATURES].apply(pd.to_numeric, errors="coerce")
    x_test = test[FEATURES].apply(pd.to_numeric, errors="coerce")
    if x_train.isna().any().any():
        raise ValueError("Missing or non-numeric values detected in training features.")
    if x_test.isna().any().any():
        raise ValueError("Missing or non-numeric values detected in test features.")

    y_train = pd.to_numeric(train["median_pIC50"], errors="coerce")
    y_test = pd.to_numeric(test["median_pIC50"], errors="coerce")
    if y_train.isna().any() or y_test.isna().any():
        raise ValueError("Target contains missing or non-numeric median_pIC50 values.")

    print("\nTraining Random Forest...")
    print(f"Trees:              {N_ESTIMATORS}")
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=1,
        max_features=1.0,
        min_samples_leaf=1,
    )
    model.fit(x_train, y_train)

    print("\nGenerating predictions...")
    predictions = model.predict(x_test)
    metrics = calculate_metrics(y_test, predictions)

    print("\n" + "=" * 60)
    print("COMBINED MODEL RESULTS")
    print("=" * 60)
    print(f"MAE:        {metrics['mae']:.4f}")
    print(f"RMSE:       {metrics['rmse']:.4f}")
    print(f"R²:         {metrics['r2']:.4f}")
    print(f"Pearson r:  {metrics['pearson_r']:.4f}")

    metrics.update(
        {
            "model": "Random Forest - combined descriptors + Morgan",
            "feature_set": "12 molecular descriptors + 2048-bit Morgan fingerprints",
            "split": "random_80_20",
            "n_descriptors": len(DESCRIPTORS),
            "n_morgan_bits": len(MORGAN_FEATURES),
            "n_features": len(FEATURES),
            "n_train": len(train),
            "n_test": len(test),
            "n_estimators": N_ESTIMATORS,
            "random_state": RANDOM_STATE,
        }
    )

    metrics_dir = OUTPUT_DIR / "metrics"
    figure_dir = OUTPUT_DIR / "figures" / "combined"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    predictions_output = test[[ID_COLUMN, "canonical_smiles", "median_pIC50"]].copy()
    predictions_output["predicted_pIC50"] = predictions
    predictions_output["residual"] = predictions_output["median_pIC50"] - predictions
    predictions_output.to_csv(metrics_dir / "combined_random_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(metrics_dir / "combined_random_metrics.csv", index=False)

    pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    ).to_csv(metrics_dir / "combined_feature_importance.csv", index=False)

    figure, axis = plt.subplots(figsize=(7, 7))
    axis.scatter(y_test, predictions, alpha=0.45, s=18)
    lower = min(float(y_test.min()), float(predictions.min()))
    upper = max(float(y_test.max()), float(predictions.max()))
    axis.plot([lower, upper], [lower, upper], linestyle="--")
    axis.set_xlabel("Observed median pIC50")
    axis.set_ylabel("Predicted pIC50")
    axis.set_title("Combined model: predicted vs observed")
    figure.tight_layout()
    figure.savefig(figure_dir / "combined_random_predicted_vs_observed.png", dpi=300)
    plt.close(figure)

    print(f"\nResults saved to: {metrics_dir}")
    print("\n" + "=" * 60)
    print("COMBINED MODEL COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
