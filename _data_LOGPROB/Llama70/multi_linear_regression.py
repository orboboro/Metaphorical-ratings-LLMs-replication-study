import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from scipy.stats import norm

import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


human_files = {
    'MB': 'human_datasets/human_MB.csv',
    'ME': 'human_datasets/human_ME.csv',
    'MI': 'human_datasets/human_MI.csv',
    'MM': 'human_datasets/human_MM.csv',
    'BA': 'human_datasets/human_BA.csv'
}

synthetic_files = {
    'MB': 'synthetic_datasets/synthetic_MB.csv',
    'ME': 'synthetic_datasets/synthetic_ME.csv',
    'MI': 'synthetic_datasets/synthetic_MI.csv',
    'MM': 'synthetic_datasets/synthetic_MM.csv',
    'BA': 'synthetic_datasets/synthetic_BA.csv'
}

datasets = ['MB','ME','MI','MM','BA']
kinds =  [human_files, synthetic_files]

def fix_commas(df):
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.replace(",", ".", regex=False)
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df


for kind in kinds:
    for dataset in datasets:
        df = pd.read_csv(kind[dataset]).dropna()
        if "synthetic" in dataset:
            df = df.drop(columns=["metaphor", "annotator"])
        else:
            df = df.drop(columns=["metaphor"])
        df = fix_commas(df)
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix ' + dataset[:-4])
        plt.show()

        