import numpy as np


def accuracy(t_hat, t_tr):
    return np.sum(t_hat == t_tr) / len(t_tr)


def F1score(prec, rec):
    return 2 * (prec*rec)/(prec + rec)


def F_neg_score(TN, FN, FP):
    return TN/(TN + (FP+FN)/2)


def cnf_mtx(t_hat, t_true):
    t_true = np.array(t_true)
    t_hat = np.array(t_hat)

    tp = np.sum((t_true == 1) & (t_hat == 1))
    fp = np.sum((t_true == 0) & (t_hat == 1))
    tn = np.sum((t_true == 0) & (t_hat == 0))
    fn = np.sum((t_true == 1) & (t_hat == 0))

    return tp, fp, tn, fn


def precision(TP, FP):
    return TP / (TP + FP)


def recall(TP, FN):
    return TP / (TP + FN)


def true_positive_rate(TP, FN):
    return TP / (TP + FN)


def false_positive_rate(FP, FN):
    return FP / (FP + FN)
