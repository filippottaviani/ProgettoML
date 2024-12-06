import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


def fixed_validation_setPer(train_x, test_x, train_t, test_t):
    scorePer = 0
    best_pen = 'solver'

    modelPerNP = Perceptron()  # nessuna penalità
    modelPerNP.fit(train_x, train_t)
    scorePerNP = modelPerNP.score(test_x, test_t)

    modelPerL2 = Perceptron(penalty='l2')
    modelPerL2.fit(train_x, train_t)
    scorePerL2 = modelPerL2.score(test_x, test_t)

    modelPerL1 = Perceptron(penalty='l1')
    modelPerL1.fit(train_x, train_t)
    scorePerL1 = modelPerL1.score(test_x, test_t)

    modelPerEN = Perceptron(penalty='elasticnet')
    modelPerEN.fit(train_x, train_t)
    scorePerEN = modelPerEN.score(test_x, test_t)

    scores = [scorePerNP, scorePerL2, scorePerL1, scorePerEN]

    if np.argmax(scores) == 0:
        scorePer = scorePerNP
        best_pen = 'None'
    elif np.argmax(scores) == 1:
        scorePer = scorePerL2
        best_pen = 'l2'
    elif np.argmax(scores) == 2:
        scorePer = scorePerL1
        best_pen = 'l1'
    else:
        scorePer = scorePerEN
        best_pen = 'elasticnet'

    return scorePer, best_pen


def fixed_validation_setLR(train_x, test_x, train_t, test_t, hyp):
    max_score = 0
    best_hyp = -1

    for i in range(len(hyp)):
        model = LogisticRegression(C=hyp[i])
        model.fit(train_x, train_t)
        scoreLR = model.score(test_x, test_t)

        if scoreLR > max_score:
            max_score = scoreLR
            best_hyp = hyp[i]
        else: pass

    return max_score, best_hyp


def fixed_validation_setSVM(train_x, test_x, train_t, test_t, hyp):
    max_score = 0
    best_hyp = -1

    for i in range(len(hyp)):
        model = svm.SVC(C=hyp[i])
        model.fit(train_x, train_t)
        scoreSVM = model.score(test_x, test_t)

        if scoreSVM > max_score:
            max_score = scoreSVM
            best_hyp = hyp[i]
        else: pass

    return max_score, best_hyp


def fixed_validation_setKNN(train_x, test_x, train_t, test_t):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(train_x, train_t)
    scoreKNN = model.score(test_x, test_t)

    return scoreKNN


def fixed_validation_setNB(train_x, test_x, train_t, test_t):
    model = GaussianNB()
    model.fit(train_x, train_t)
    scoreNB = model.score(test_x, test_t)

    return scoreNB


def fixed_validation_setLDA(train_x, test_x, train_t, test_t):
    scoreLDA = 0
    best_solver = 'solver'

    modelsvd = LinearDiscriminantAnalysis(solver='svd')
    modelsvd.fit(train_x, train_t)
    scoreLDAsvd = modelsvd.score(test_x, test_t)

    modellsqr = LinearDiscriminantAnalysis(solver='lsqr')
    modellsqr.fit(train_x, train_t)
    scoreLDAlsqr = modellsqr.score(test_x, test_t)

    modeleigen = LinearDiscriminantAnalysis(solver='eigen')
    modeleigen.fit(train_x, train_t)
    scoreLDAeigen = modeleigen.score(test_x, test_t)

    scores = [scoreLDAsvd, scoreLDAlsqr, scoreLDAeigen]

    if np.argmax(scores) == 0:
        scoreLDA = scoreLDAsvd
        best_solver = 'svd'
    elif np.argmax(scores) == 1:
        scoreLDA = scoreLDAlsqr
        best_solver = 'lsqr'
    else:
        scoreLDA = scoreLDAeigen
        best_solver = 'eigen'

    return scoreLDA, best_solver


def fixed_validation_setQDA(train_x, test_x, train_t, test_t):
    model = QuadraticDiscriminantAnalysis()
    model.fit(train_x, train_t)
    scoreQDA = model.score(test_x, test_t)

    return scoreQDA


def fixed_validation_setMLP(train_x, test_x, train_t, test_t, hyp):
    max_score = 0
    best_hyp = -1

    for i in range(len(hyp)):

        model = MLPClassifier(max_iter=700, alpha=hyp[i])
        model.fit(train_x, train_t)
        scoreNN = model.score(test_x, test_t)

        if scoreNN > max_score:
            max_score = scoreNN
            best_hyp = hyp[i]
        else: pass

    return max_score, best_hyp


def ms_graph(scores):
    data = {
        "Model": ["Perceptron", "Logistic Regression", "Support Vector Machine", "K-Nearest Neighbor", "Naive Bayes",
                  "Linear Discriminant Analysis", "Quadratic Discriminant Analysis", "Multi-Layer Perceptron"],
        "Scores": [scores[0], scores[1], scores[2], scores[3], scores[4], scores[5], scores[6], scores[7]],
        "Optimized": [True, True, True, False, False, True, False, True]
    }
    models = pd.DataFrame(data)

    plt.figure(figsize=(22, 10), dpi=80)
    plt.title('Risultati della Model Selection')
    sns.barplot(data=models, x="Model", y="Scores", palette=['#c26f4f', '#52627f'], hue="Optimized")
    plt.show()

