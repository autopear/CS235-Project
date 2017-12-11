import os
import random
import itertools
import time
import numpy as np
import scipy.sparse

from scipy.sparse import csr_matrix
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC


# Returns training and validation labels and data
def load_data(fold_combination):
    ''' fold_combination: a list of integers indicating the fold '''
    fold_combination = sorted(fold_combination)

    lsi_folder_path = os.path.join("tf-idf_data", "lsi")
    fold_str = ''.join(["_" + str(fold) for fold in fold_combination])

    lsi_train_path = os.path.join(lsi_folder_path, "lsi_fold" + fold_str + ".npy")
    lsi_valid_path = os.path.join(lsi_folder_path, "lsi_valid" + fold_str + ".npy")

    lsi_train = np.load(lsi_train_path)
    lsi_valid = np.load(lsi_valid_path)

    train_labels = lsi_train[:, 0]
    train_labels = np.reshape(train_labels, (-1, 1))

    valid_labels = lsi_valid[:, 0]
    valid_labels = np.reshape(valid_labels, (-1, 1))

    lsi_train = lsi_train[:, 1:]
    lsi_valid = lsi_valid[:, 1:]

    return train_labels, lsi_train, valid_labels, lsi_valid


def get_precision(rates):
    tp = rates[0]
    fp = rates[1]
    return tp / (tp + fp + 1)


def get_recall(rates):
    tp = rates[0]
    fn = rates[3]
    return tp / (tp + fn + 1)


def get_specificity(rates):
    tn = rates[2]
    fp = rates[1]
    return tn / (fp + tn + 1)


def get_fscore(precision, recall):
    return 2 * (precision * recall/(precision+recall + 1))


def get_accuracy(rates):
    tp = rates[0]
    fp = rates[1]
    tn = rates[2]
    fn = rates[3]
    return (tp + tn) / (tp + fp + tn + fn)


if __name__ == "__main__":
    all_folds = [1, 2, 3, 4, 5]
    all_folds = set(all_folds)
    lambd = 1.5e-4

    sv_folder_path = os.path.join("tf-idf_data", "sv")
    if not os.path.exists(sv_folder_path):
        os.makedirs(sv_folder_path)

    # element is tuple (tp, fp, tn, fn)
    linear_rates = []
    rbf_rates = []

    for fold in itertools.combinations(all_folds, 4):
        fold = list(fold)
        fold_str = "_".join(map(str, fold))
        train_labels, lsi_train, valid_labels, lsi_valid = load_data(fold)

        linear_SVC = SVC(kernel="linear", max_iter=6000)
        rbf_SVC = SVC(kernel="rbf", max_iter=6000)

        linear_SVC.fit(lsi_train, train_labels.ravel())
        rbf_SVC.fit(lsi_train, train_labels.ravel())

        linear_predictions = linear_SVC.predict(lsi_valid)
        rbf_predictions = rbf_SVC.predict(lsi_valid)

        tn, fp, fn, tp = confusion_matrix(valid_labels, linear_predictions).ravel()
        linear_rates.append((tp, fp, tn, fn))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        fscore = 2 * ((precision * recall) / (precision + recall))
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        print("precision:", precision,
              "recall:", recall,
              "specificity:", specificity,
              "fscore:", fscore,
              "accuracy:", accuracy)
        print("Linear TP/FP/TN/FN rate for fold", fold_str, ":", tp, fp, tn, fn)
        print("--" * 50)
        tn, fp, fn, tp = confusion_matrix(valid_labels, rbf_predictions).ravel()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        fscore = 2 * ((precision * recall) / (precision + recall))
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        print("precision:", precision,
              "recall:", recall,
              "specificity:", specificity,
              "fscore:", fscore,
              "accuracy:", accuracy)
        rbf_rates.append((tp, fp, tn, fn))
        print("RBF TP/FP/TN/FN rate for fold", fold_str, ":", tp, fp, tn, fn)
        print("--" * 50)

    agg_precision = []
    agg_recall = []
    agg_specificity = []
    agg_fscore = []
    agg_accuracy = []
    for rates in linear_rates:
        precision = get_precision(rates)
        recall = get_recall(rates)
        specificity = get_specificity(rates)
        fscore = get_fscore(precision, recall)
        accuracy = get_accuracy(rates)

        agg_precision.append(precision)
        agg_recall.append(recall)
        agg_specificity.append(specificity)
        agg_fscore.append(fscore)
        agg_accuracy.append(accuracy)

    avg_precision = sum(agg_precision) / len(agg_precision)
    avg_recall = sum(agg_recall) / len(agg_recall)
    avg_specificity = sum(agg_specificity) / len(agg_specificity)
    avg_fscore = sum(agg_fscore) / len(agg_fscore)
    avg_accuracy = sum(agg_accuracy) / len(agg_accuracy)

    print(avg_precision, avg_recall, avg_specificity, avg_fscore, avg_accuracy)
    print("--" * 50)

    agg_precision = []
    agg_recall = []
    agg_specificity = []
    agg_fscore = []
    agg_accuracy = []
    for rates in rbf_rates:
        precision = get_precision(rates)
        recall = get_recall(rates)
        specificity = get_specificity(rates)
        fscore = get_fscore(precision, recall)
        accuracy = get_accuracy(rates)

        agg_precision.append(precision)
        agg_recall.append(recall)
        agg_specificity.append(specificity)
        agg_fscore.append(fscore)
        agg_accuracy.append(accuracy)

    avg_precision = sum(agg_precision) / len(agg_precision)
    avg_recall = sum(agg_recall) / len(agg_recall)
    avg_specificity = sum(agg_specificity) / len(agg_specificity)
    avg_fscore = sum(agg_fscore) / len(agg_fscore)
    avg_accuracy = sum(agg_accuracy) / len(agg_accuracy)

    print(avg_precision, avg_recall, avg_specificity, avg_fscore, avg_accuracy)
    print("--" * 50)


