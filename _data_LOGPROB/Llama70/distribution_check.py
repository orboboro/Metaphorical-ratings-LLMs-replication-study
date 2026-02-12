import pandas as pd
from scipy.stats import shapiro
import os
import matplotlib.pyplot as plt
import seaborn as sns

human_path = "human_datasets/"
synthetic_path = "synthetic_datasets/"

output_path = "output/"
os.makedirs(output_path, exist_ok=True)
os.makedirs(os.path.join(output_path, "plots"), exist_ok=True)

# -------------------------
# FILE LIST
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
    results = []
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

        results.append({
            "dimension": dim,
            "N": len(values),
            "W": stat,
            "p_value": p,
            "normal": p > 0.05
        })

    return pd.DataFrame(results)

shapiro_human_df = run_tests(human_values, "HUMAN")
shapiro_synth_df = run_tests(synthetic_values, "SYNTHETIC")

# -------------------------
# SAVE VALUES CSV
# -------------------------

# Human
human_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in human_values.items()]))
human_csv_path = os.path.join(output_path, "human_values.csv")
human_df.to_csv(human_csv_path, index=False)

# Synthetic
synthetic_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in synthetic_values.items()]))
synthetic_csv_path = os.path.join(output_path, "synthetic_values.csv")
synthetic_df.to_csv(synthetic_csv_path, index=False)

print(f"\nValues saved to {human_csv_path} and {synthetic_csv_path}")

# -------------------------
# PLOT DISTRIBUTIONS
# -------------------------

def plot_distributions(values_dict, label):
    for dim, values in values_dict.items():
        plt.figure(figsize=(6,4))
        sns.histplot(values, kde=True, bins=15, color="skyblue")
        plt.title(f"{dim} ({label})")
        plt.xlabel(dim)
        plt.ylabel("Count")
        plt.tight_layout()

        plot_path = os.path.join(output_path, "plots", f"{dim}_{label}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot: {plot_path}")

plot_distributions(human_values, "HUMAN")
plot_distributions(synthetic_values, "SYNTHETIC")
