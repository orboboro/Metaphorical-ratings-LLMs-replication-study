import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Percorsi cartelle
human_folder = "human_datasets"
synthetic_folder = "synthetic_datasets"
plot_folder = "plots"

os.makedirs(plot_folder, exist_ok=True)

# Lista dei dataset umani e sintetici
human_files = [f for f in os.listdir(human_folder) if f.endswith(".csv")]
synthetic_files = [f for f in os.listdir(synthetic_folder) if f.endswith(".csv")]

# Funzione per caricare e normalizzare i dataset
def load_human_file(path):
    df = pd.read_csv(path)
    # sostituisci le virgole con i punti per i numeri decimali e converti a float
    for col in df.columns:
        if "human" in col:
            df[col] = df[col].astype(str).str.replace(",", ".").astype(float)
    return df

def load_synthetic_file(path):
    df = pd.read_csv(path)
    # converti i valori numerici a float
    for col in df.columns:
        if "synthetic" in col:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Dizionari per aggregare i dati per dimensione
human_data = {}
synthetic_data = {}

# Carica i dati umani
for file in human_files:
    df = load_human_file(os.path.join(human_folder, file))
    for col in df.columns:
        if "_human" in col:
            dim = col.replace("_human","")
            if dim not in human_data:
                human_data[dim] = []
            human_data[dim].extend(df[col].dropna().tolist())

# Carica i dati sintetici
for file in synthetic_files:
    df = load_synthetic_file(os.path.join(synthetic_folder, file))
    for col in df.columns:
        if "_synthetic" in col:
            dim = col.replace("_synthetic","")
            if dim not in synthetic_data:
                synthetic_data[dim] = []
            synthetic_data[dim].extend(df[col].dropna().tolist())

# Crea i grafici
for dim in human_data.keys():
    plt.figure(figsize=(8,5))
    sns.kdeplot(human_data[dim], label="Human", color="blue", fill=True, alpha=0.3)
    if dim in synthetic_data:
        sns.kdeplot(synthetic_data[dim], label="Synthetic", color="red", fill=True, alpha=0.3)
    plt.title(f"Distribuzione dei rating - {dim}")
    plt.xlabel("Valore del rating")
    plt.ylabel("Densità")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, f"{dim}_distribution.png"))
    plt.close()

print(f"Grafici salvati in '{plot_folder}'")
