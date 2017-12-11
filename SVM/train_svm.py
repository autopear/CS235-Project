import os
import random
import itertools
import time
import numpy as np
import scipy.sparse

from scipy.sparse import csr_matrix
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import confusion_matrix


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


def pegasos_linear_fit(data, labels, lambd, iterations=50000):
    # add bias term to data
    bias = np.ones((data.shape[0], 1))
    data = np.hstack([bias, data])

    w = np.zeros((1, data.shape[1]))
    for t in range(1, iterations):
        idx = random.sample(range(0, data.shape[0]), 1)
        eta = 1 / (lambd * t)

        indicator_fn = labels[idx] * np.dot(data[idx, :], w.T)
        if indicator_fn < 1:
            w = (1 - eta * lambd) * w + (eta * labels[idx] * data[idx, :])
        else:
            w = (1 - eta * lambd) * w

    return w


def pegasos_rbf_fit(data, labels, lambd, iterations=50000, sigma=1):
    # add bias term to data
    bias = np.ones((data.shape[0], 1))
    data = np.hstack([bias, data])

    alpha = np.zeros((1, data.shape[0]))
    kernel = RBF(length_scale=sigma)
    for t in range(1, iterations):
        idx = random.sample(range(0, data.shape[0]), 1)
        eta = 1 / (lambd * t)

        indicator_fn = 0
        for j in range(0, data.shape[0]):
            if j != idx:
                indicator_fn += alpha[0, j] * labels[idx] * kernel(
                    np.reshape(data[idx, :], (-1, data.shape[1])), np.reshape(data[j, :], (-1, data.shape[1])))

        indicator_fn = labels[idx] * eta * indicator_fn

        # alpha[idx] is the weight of a support vector
        if indicator_fn < 1:
            alpha[0, idx] = alpha[0, idx] + 1

    return alpha


def svm_linear_predict(unseen_data, w):
    # add bias term to data
    bias = np.ones((unseen_data.shape[0], 1))
    unseen_data = np.hstack([bias, unseen_data])

    predictions = np.dot(unseen_data, w.T)
    predictions[predictions >= 0] = 1
    predictions[predictions < 0] = -1

    return predictions


def svm_rbf_predict(unseen_data, data, labels, alpha, sigma=1):
    # add bias term to data
    unseen_bias = np.ones((unseen_data.shape[0], 1))
    unseen_data = np.hstack([unseen_bias, unseen_data])
    bias = np.ones((data.shape[0], 1))
    data = np.hstack([bias, data])
    kernel = RBF(length_scale=sigma)

    predictions = np.dot(kernel(unseen_data, data), np.multiply(alpha.T, labels))

    predictions[predictions >= 0] = 1
    predictions[predictions < 0] = -1

    return predictions


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

        start_time = time.time()
        weights = pegasos_linear_fit(lsi_train, train_labels, lambd=lambd, iterations=6000)
        print("Time for fitting fold", fold_str, " using linear kernel: ", time.time() - start_time)

        start_time = time.time()
        alphas = pegasos_rbf_fit(lsi_train, train_labels, lambd=lambd, iterations=6000)
        print("Time for fitting fold", fold_str, " using RBF kernel: ", time.time() - start_time)

        weight_path = os.path.join(sv_folder_path, "weight_" + fold_str)
        alpha_path = os.path.join(sv_folder_path, "alpha_" + fold_str)

        np.save(weight_path, weights)
        np.save(alpha_path, alphas)

        linear_predictions = svm_linear_predict(lsi_valid, weights)
        rbf_predictions = svm_rbf_predict(lsi_valid, lsi_train, train_labels, alphas)

        tn, fp, fn, tp = confusion_matrix(valid_labels, linear_predictions).ravel()
        linear_rates.append((tp, fp, tn, fn))
        print("Linear TP/FP/TN/FN rate for fold", fold_str, ":", tp, fp, tn, fn)

        tn, fp, fn, tp = confusion_matrix(valid_labels, rbf_predictions).ravel()
        rbf_rates.append((tp, fp, tn, fn))
        print("RBF TP/FP/TN/FN rate for fold", fold_str, ":", tp, fp, tn, fn)


    for x in linear_rates:
        print(x)

    for x in rbf_rates:
        print(x)


