import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import auc, roc_curve, confusion_matrix

from Metrics import cnf_mtx, accuracy, recall, precision, true_positive_rate, false_positive_rate, F_neg_score, F1score


def model_training(scores, hyps, X_tr, t_tr):
    if np.argmax(scores) == 0:
        model = Perceptron()
        model_name = "Perceptron"
    elif np.argmax(scores) == 1:
        model = LogisticRegression()
        model_name = "Linear Regression"
    elif np.argmax(scores) == 2:
        model = svm.SVC(C=hyps[2])
        model_name = "Support Vector Machine"
    elif np.argmax(scores) == 3:
        model = KNeighborsClassifier(n_neighbors=5)
        model_name = "K-Nearest Neighbor"
    elif np.argmax(scores) == 4:
        model = GaussianNB()
        model_name = "Naive Bayes"
    elif np.argmax(scores) == 5:
        model = LinearDiscriminantAnalysis()
        model_name = "Linear Discriminant Analysis"
    elif np.argmax(scores) == 6:
        model = QuadraticDiscriminantAnalysis()
        model_name = "Quadratic Discriminant Analysis"
    else:
        model = MLPClassifier(max_iter=1000, alpha=hyps[7])
        model_name = "Multi-Layer Perceptron (Neural Network)"

    model.fit(X_tr, t_tr)  # training

    return model, model_name


def test_model(model, model_name, X_te, t_te):
    t_hat = model.predict(X_te)
    acc = accuracy(t_hat, t_te)
    print("\nAccuratezza di ", model_name," sul test set: ", acc)

    # metriche per la classificazione
    tp, fp, tn, fn = cnf_mtx(t_hat, t_te)
    print("\nMetriche per la Confusion Matrix:\nTrue positive:", tp, "\nTrue negative:", tn, "\nFalse positive:", fp, "\nFalse negative:", fn)

    # precision e recall
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    prec_n = precision(tn, fn)
    rec_n = recall(tn, fp)
    print("\nPrecision: ", prec, "\nRecall: ", rec, "\nPrecision (negativo): ", prec_n, "\nRecall (negativo): ", rec_n)

    # f1 score
    f1score = F1score(prec, rec)
    f_neg = F_neg_score(tn,fn, fp)
    print("\nF1 score: ", f1score, "\nF1 score (negativo): ", f_neg)

    # True Positive Rate e False Positive Rate
    TPR = true_positive_rate(tp, fn)
    FPR = false_positive_rate(fp, fn)
    TNR = true_positive_rate(tn, fp)
    FNR = false_positive_rate(fn, fp)
    print("\nTrue Positive Rate (TPR):", TPR, "\nFalse Positive Rate (FPR):", FPR,"\nTrue Negative Rate (TNR):", TNR, "\nFalse Negative Rate (FNR):", FNR)

    # confusion matrix e ROC
    plt.figure(figsize=(18, 9), dpi=80)
    plt.subplot(1,2,1)
    plt.title('Confusion Matrix')
    cm = confusion_matrix(y_true=t_te, y_pred=t_hat, normalize='true')
    sns.heatmap(cm, annot=True,  xticklabels= ["Non potabile", "Potabile"], yticklabels= ["Non potabile", "Potabile"])
    plt.subplot(1, 2, 2)

    if model_name != "Multi-Layer Perceptron (Neural Network)":
        fpr, tpr, _ = roc_curve(t_te, model.decision_function(X_te))
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend()
        plt.tight_layout()
        plt.show()