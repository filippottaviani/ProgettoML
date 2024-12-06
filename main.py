import pandas as pd
from sklearn.metrics import precision_recall_curve, confusion_matrix, RocCurveDisplay,  roc_curve, auc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
from sklearn.preprocessing import minmax_scale
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from EDAPrep import data_split, densities, visualize_data, standardization
from ModelSelection import fixed_validation_setLR, fixed_validation_setSVM, fixed_validation_setKNN, fixed_validation_setNB, fixed_validation_setLDA, fixed_validation_setQDA, fixed_validation_setPer, fixed_validation_setMLP, ms_graph
from sklearn.neural_network import MLPClassifier
from ModelAssessment import model_training, test_model


# ispezione del dataframe
print("\n####################################### EDA PHASE #######################################\n")

# importazione del dataset
data = 'dataset/water_potability.csv'
df = pd.read_csv(data)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print("\nDataset importato\n\n", df.describe(), "\n\nSono stati rilevati campioni con almeno una feature nulla:\n", df.isnull().sum())

# elimino campioni con
col_to_drop = []
df = df.drop(columns=col_to_drop) # eliminate dopo il calcolo VIF
df.dropna(inplace = True)

# Matrice di correlazione e sbilanciamento
visualize_data(df)

# Densità delle features
densities(df)

# pair plot
#sns.pairplot(data=df, hue="Potability", palette='pastel')
#plt.show()

# individuazione di features ed etichette
X = df.drop(columns = 'Potability')
t = df['Potability'].values
print("\nIl dataset è stato separato in:\nDesign matrix: ", X.shape,"\nEtichette: ", t.shape)

# sbilanciamento del dataset
print("\nPer controllare l'eventuale sbilanciamento del dataset verifico le proporzioni delle etichette:")
print("Campioni potabili: ", np.sum(t == 1), "  ",(np.sum(t == 1)/ (t.shape[0]))*100," %")
print("Campioni non potabili: ",  np.sum(t == 0), "  ",(np.sum(t == 0)/ (t.shape[0]))*100," %\n")

print("\n##################################### PREPROCESSING #####################################")
# divisione del dataset
X_tr, X_te, t_tr, t_te = train_test_split(X, t, train_size=0.7, test_size=0.3, random_state=100)
print("\nNumero di campioni per set dopo la divisione:\nTraning set: ", X_tr.shape[0], "\nTest set: ", X_te.shape[0], "\nTotale:", X.shape[0])

# variance inflation factor
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(X_tr, i) for i in range(X_tr.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
print("\nVariance Inflation Factor (VIF): \n", vif.T, "\n")

# normalizzazione
X_tr = minmax_scale(X_tr, axis=0)
X_te = minmax_scale(X_te, axis=0)
#X_tr, X_te = standardization(X_tr, X_te) # vanno usate media e variazione standard del training set

print("La normalizzazione delle feature è di tipo Min-Max Scaling ed ha lo scopo di migliorare i \ntempi di elaborazione.\n")

# model selection
print("\n#################################### MODEL SELECTION ####################################")

train_x, val_x, train_t, val_t = train_test_split(X_tr, t_tr, train_size=0.9, test_size=0.1)
hyp = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
print("\nNumero di campioni per Model Selection:\nTraning set: ", train_x.shape[0],"  ", (train_x.shape[0]/X_tr.shape[0])*100," %" ,"\nValidation set: ", val_x.shape[0],"  ", (val_x.shape[0]/X_tr.shape[0])*100," %")

# Perceptron
scorePer, best_pen = fixed_validation_setPer(train_x, val_x, train_t, val_t)
print("\nPerceptron\nScore:", scorePer,"\nPenalità migliore: ", best_pen)

# Logistic Regression
scoreLR, best_hypLR = fixed_validation_setLR(train_x, val_x, train_t, val_t, hyp)
hyp1 = [0.2*best_hypLR, 0.4*best_hypLR, 0.6*best_hypLR, 0.8*best_hypLR, best_hypLR, 2*best_hypLR, 4*best_hypLR, 6*best_hypLR, 8*best_hypLR]
scoreLR1, best_hypLR1 = fixed_validation_setLR(train_x, val_x, train_t, val_t, hyp1)
print("\nLogistic Regression\nScore:", scoreLR1,"\nIperparametro migliore:",best_hypLR1)

# Support Vector Machine
scoreSVM, best_hypSVM = fixed_validation_setSVM(train_x, val_x, train_t, val_t, hyp)
hyp1 = [0.2*best_hypSVM, 0.4*best_hypSVM, 0.6*best_hypSVM, 0.8*best_hypSVM, best_hypSVM, 2*best_hypSVM, 4*best_hypSVM, 6*best_hypSVM, 8*best_hypSVM]
scoreSVM1, best_hypSVM1 = fixed_validation_setSVM(train_x, val_x, train_t, val_t, hyp1)
print("\nSupport Vector Machine\nScore:", scoreSVM1,"\nIperparametro migliore:", best_hypSVM1)

# K-Nearest Neighbor
scoreKNN = fixed_validation_setKNN(train_x, val_x, train_t, val_t)
print("\nKNN Classifier\nScore:", scoreKNN,"\nIperparametro migliore: non ottimizzabile")

# Naive Bayes
scoreNB = fixed_validation_setNB(train_x, val_x, train_t, val_t)
print("\nNaive Bayes\nScore:", scoreNB,"\nIperparametro migliore: non ottimizzabile")

# Linear Discriminant Analysis
scoreLDA, best_solverLDA = fixed_validation_setLDA(train_x, val_x, train_t, val_t)
print("\nLinear Discriminant Analysis\nScore:", scoreLDA,"\nSolver migliore: ", best_solverLDA)

# Quadratic Discriminant Analysis
scoreQDA = fixed_validation_setQDA(train_x, val_x, train_t, val_t)
print("\nQuadratic Discriminant Analysis\nScore:", scoreQDA,"\nIperparametro migliore: non ottimizzabile")

# Multi-Layer Perceptron (Neural Network)
scoreMLP, best_hypMLP = fixed_validation_setMLP(train_x, val_x, train_t, val_t, hyp)
hyp1 = [0.2*best_hypMLP, 0.4*best_hypMLP, 0.6*best_hypMLP, 0.8*best_hypMLP, best_hypMLP, 2*best_hypMLP, 4*best_hypMLP, 6*best_hypMLP, 8*best_hypMLP]
scoreMLP1, best_hypMLP1 = fixed_validation_setMLP(train_x, val_x, train_t, val_t, hyp1)
print("\nMulti-Layer Perceptron (Neural Network)\nScore:", scoreMLP1,"\nIperparametro migliore: ", best_hypMLP1)

# Grafico gli score dei modelli
scores = [scorePer, scoreLR1, scoreSVM1, scoreKNN, scoreNB, scoreLDA, scoreQDA, scoreMLP1]
hyps = [0, 0, best_hypSVM1, 0, 0, 0, 0, best_hypMLP1]
ms_graph(scores)

print("\n\n###################################### MODEL ASSESSMENT ######################################")

best_model, model_name = model_training(scores, hyps, X_tr, t_tr)
print("\nLa fase di test sarà incentrata sul modello: ", model_name)

test_model(best_model, model_name, X_te, t_te)
print("\nFase di test completata")
