import torch
from torch import logical_and, logical_or, sum


def Accuracy(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        p = sum(logical_and(y_true[i], y_pred[i]))
        q = sum(logical_or(y_true[i], y_pred[i]))
        count += p / q
    return count / y_true.shape[0]

def Precision(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        if sum(y_pred[i]) == 0:
            continue
        count += sum(logical_and(y_true[i], y_pred[i])) / sum(y_pred[i])
    return count / y_true.shape[0]

def Recall(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        if sum(y_true[i]) == 0:
            continue
        count += sum(logical_and(y_true[i], y_pred[i])) / sum(y_true[i])
    return count / y_true.shape[0]


def F1Measure(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        if (sum(y_true[i]) == 0) and (sum(y_pred[i]) == 0):
            continue
        p = sum(logical_and(y_true[i], y_pred[i]))
        q = sum(y_true[i]) + sum(y_pred[i])
        count += (2 * p) / q
    return count / y_true.shape[0]
