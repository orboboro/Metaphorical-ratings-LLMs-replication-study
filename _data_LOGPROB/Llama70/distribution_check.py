import pandas as pd
from scipy.stats import shapiro
import os

human_path = "human_datasets/"
synthetic_path = "synthetic_datasets/"

# -------------------------
# FILE LIST (no glob)
# -------------------------

human_files = [
    "human_BA.csv",
    "human_MB.csv",
    "human_ME.csv",
    "human_MI.csv",
    "human_MM.csv"
]

synthetic_files = [
    "synthetic_BA.csv",
    "synthetic_MB.csv",
    "synthetic_ME.csv",
    "synthetic_MI.csv",
    "synthetic_MM.csv"
]

# -------------------------
# COLLECTORS
# -------------------------

human_values = {}
synthetic_values = {}

# -------------------------
# HUMAN DATA LOADING
# -------------------------

for fname in human_files:
    path = os.path.join(human_path, fname)
    df = pd.read_csv(path)

    for col in df.columns:
        if col == "metaphor":
            continue

        # convert comma decimals to float
        series = (
            df[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
        )

        series = pd.to_numeric(series, errors="coerce").dropna()

        human_values.setdefault(col, []).extend(series.tolist())

# -------------------------
# SYNTHETIC DATA LOADING
# -------------------------

for fname in synthetic_files:
    path = os.path.join(synthetic_path, fname)
    df = pd.read_csv(path)

    for col in df.columns:
        if col in ["metaphor", "annotator"]:
            continue

        series = pd.to_numeric(df[col], errors="coerce").dropna()
        synthetic_values.setdefault(col, []).extend(series.tolist())

# -------------------------
# SHAPIRO TEST
# -------------------------

print("\n===== SHAPIRO–WILK NORMALITY TEST =====\n")

def run_tests(values_dict, label):
    print(f"\n--- {label} ---\n")
    for dim, values in sorted(values_dict.items()):
        if len(values) < 3:
            print(f"{dim}: not enough data")
            continue

        stat, p = shapiro(values)

        print(f"{dim}")
        print(f"  N = {len(values)}")
        print(f"  W = {stat:.4f}")
        print(f"  p = {p:.6f}")
        print(f"  normal? {'YES' if p > 0.05 else 'NO'}\n")


run_tests(human_values, "HUMAN")
run_tests(synthetic_values, "SYNTHETIC")
