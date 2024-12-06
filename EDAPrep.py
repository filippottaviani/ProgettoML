import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def standardization(X_tr, X_te):
    tr_mean = X_tr.mean()
    tr_std = X_tr.std()

    norm_tr = (X_tr - tr_mean) / tr_std
    norm_te = (X_te - tr_mean) / tr_std

    return norm_tr, norm_te


def data_split(data, tr, val):
    tr_size = int(tr * data.shape[0])
    val_size = int(val * data.shape[0])

    data_tr = data[0: tr_size]
    data_val = data[tr_size:tr_size+val_size]
    data_te = data[val_size+tr_size:]

    return data_tr, data_val, data_te


def visualize_data(df):
    plt.figure(figsize=(18, 12), dpi=80)
    plt.subplot(2, 2, 1)
    sns.countplot(data=df, x="Potability", hue="Potability", palette=['#b8464d', '#1ea877'])
    plt.title('Potabile vs Non potabile')
    plt.xlabel('Potability')
    plt.ylabel('')
    plt.subplot(2, 2, 2)
    sns.heatmap(df.corr(), annot=True)  # matrice di correlazione
    plt.title('Matrice di correlazione')
    plt.subplot(2, 1, 2)
    sns.scatterplot(data=df, x="Turbidity", y="ph", style="Potability", hue="Potability", palette=['#b8464d', '#1ea877'])  # scatterplot
    plt.tight_layout()
    plt.show()


def densities(df):
    plt.figure(figsize=(18, 12), dpi=80)
    plt.subplot(2, 2, 1)
    sns.kdeplot(data=df, x="ph", hue="Potability", multiple="stack", palette=['#b8464d', '#1ea877'])
    plt.subplot(2, 2, 2)
    sns.kdeplot(data=df, x="Hardness", hue="Potability", multiple="stack", palette=['#b8464d', '#1ea877'])
    plt.subplot(2, 2, 3)
    sns.kdeplot(data=df, x="Trihalomethanes", hue="Potability", multiple="stack", palette=['#b8464d', '#1ea877'])
    plt.subplot(2, 2, 4)
    sns.kdeplot(data=df, x="Turbidity", hue="Potability", multiple="stack", palette=['#b8464d', '#1ea877'])
    plt.tight_layout()
    plt.show()

