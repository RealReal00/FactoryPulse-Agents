from pathlib import Path

import pandas as pd


DATASET = Path(__file__).parent / "data" / "raw" / "ai4i2020.csv"
PHYSICAL = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def main():
    df = pd.read_csv(DATASET)
    print(f"Righe: {len(df)}; colonne: {len(df.columns)}")
    print("\nColonne e tipi:\n", df.dtypes.to_string())
    print("\nValori mancanti:\n", df.isna().sum().to_string())
    print(f"\nDuplicati completi: {df.duplicated().sum()}")
    print(f"Duplicati sulle sole feature fisiche: {df.duplicated(subset=PHYSICAL).sum()}")
    counts = df["Machine failure"].value_counts().sort_index()
    print("\nDistribuzione Machine failure:")
    for value, count in counts.items():
        print(f"  {value}: {count} ({count / len(df) * 100:.2f}%)")
    print("\nGuasti per tipologia:\n", df[FAILURE_MODES].sum().to_string())
    any_mode = df[FAILURE_MODES].eq(1).any(axis=1)
    target = df["Machine failure"].eq(1)
    print(f"\nTarget 1 senza modalità segnalata: {(target & ~any_mode).sum()}")
    print(f"Target 0 con modalità segnalata: {(~target & any_mode).sum()}")
    print(f"Righe con più modalità segnalate: {df[FAILURE_MODES].sum(axis=1).gt(1).sum()}")


if __name__ == "__main__":
    main()
