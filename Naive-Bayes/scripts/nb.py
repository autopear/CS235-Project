# Naive-Bayes classification


import glob
import math
import numpy
import os
import scipy
from scipy.sparse import csr_matrix


def to_system_path(path):
    """ Convert an input path to the current system style, \ for Windows, / for others """
    if os.name == "nt":
        return path.replace("/", "\\")
    else:
        return path.replace("\\", "/")


def to_standard_path(path):
    """ Convert \ to / in path (mainly for Windows) """
    return path.replace("\\", "/")


def find_label(probs):
    ret = 0
    score = 0
    for i in range(1, len(probs)):
        if probs[i] > score:
            score = probs[i]
            ret = i
    return ret


def get_binary_precision_recall_f1_spc(reals, preds, true_value):
    """ Compute precision, recall and f1-score for setting true_value as binary positive """
    bin_reals = []
    for label in reals:
        if label == true_value:
            bin_reals.append(True)
        else:
            bin_reals.append(False)
    bin_preds = []
    for label in preds:
        if label == true_value:
            bin_preds.append(True)
        else:
            bin_preds.append(False)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(0, len(bin_reals)):
        real_label = bin_reals[i]
        pred_label = bin_preds[i]
        if real_label and pred_label:
            tp += 1
        elif real_label and not pred_label:
            fn += 1
        elif not real_label and pred_label:
            fp += 1
        else:
            tn += 1
    # Avoid division by zero error
    if (tp == 0) or (tn == 0) or (fp == 0) or (fn == 0):
        tp += 1
        tn += 1
        fp += 1
        fn += 1
    p = float(tp) / (tp + fp)
    r = float(tp) / (tp + fn)
    f = 2. * p * r / (p + r)
    s = float(tn) / (tn + fp)
    return p, r, f, s


def get_precision_recall_accuracy_f1_spc(reals, preds):
    """ Compute precision, recall, accuracy and f1-score for all classes """
    ps = 0
    rs = 0
    fs = 0
    ss = 0
    for i in range(1, 6):
        p, r, f, s = get_binary_precision_recall_f1_spc(reals, preds, i)
        ps += p
        rs += r
        fs += f
        ss += s
    corrects = 0
    for i in range(0, len(reals)):
        if reals[i] == preds[i]:
            corrects += 1
    return (ps / 5), (rs / 5), (float(corrects) / len(reals)), (fs / 5), (ss / 5)


dir_path = to_standard_path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # Module directory
data_dir = "{0}/data".format(dir_path)
output_dir = "{0}/output".format(dir_path)

# Read number of folds dynamically
folds = []
for fp in glob.glob(to_system_path("{0}/freqs_class-*.npz".format(data_dir))):
    folds.append(int(os.path.basename(fp)[12:-4]))
folds.sort()

num_folds = len(folds)

# Load all review vectors and corresponding labels
freqs = [None] * num_folds
labels = [None] * num_folds
lengths = [None] * num_folds
for fold in folds:
    freqs[fold-1] = scipy.sparse.load_npz(to_system_path("{0}/freqs_class-{1}.npz".format(data_dir, fold)))

    fold_labels = []
    fold_lens = numpy.array([0, 0, 0, 0, 0])  # Number of total words per class
    with open(to_system_path("{0}/fold-{1}.tsv".format(data_dir, fold)), "r") as inf:
        for line in inf:
            line = line[:-1]
            components = line.split("\t")
            if len(components) == 2:
                s = int(components[0])
                l = int(components[1])
                fold_labels.append(s)
                fold_lens[s-1] = fold_lens[s-1] + l
    inf.close()
    labels[fold-1] = fold_labels
    lengths[fold-1] = fold_lens
    del fold_labels, fold_lens
print("Labels and number of words loaded")

# Statistics for binary version
tps = []  # True positive
tns = []  # True negative
fps = []  # False positive
fns = []  # False negative

# Statistics for multiclass version
precisions = []
recalls = []
accuracies = []
f1s = []  # F1 scores
spcs = []  # Specificities

for test_fold in folds:

    train_freqs = None
    train_lens = None

    # Construct training data
    for train_fold in folds:
        if train_fold == test_fold:
            continue

        if train_freqs is None:
            train_freqs = freqs[train_fold-1]
            train_lens = lengths[train_fold-1]
        else:
            train_freqs = train_freqs + freqs[train_fold-1]
            train_lens = train_lens + lengths[train_fold-1]
    print("Done training data")

    vs = train_freqs.shape[1]  # Vocabulary size

    probs_multi = numpy.zeros((5, vs), dtype=float)
    probs_bin = numpy.zeros((2, vs), dtype=float)  # row 0 negative, 1 positive

    len_n = train_lens[0] + train_lens[1] + train_lens[2]  # Negative lengths
    len_p = train_lens[3] + train_lens[4]  # Positive lengths

    # Use Laplacian correction and log-scaled probability
    # Log-scaled can deal with small float numbers, and the final probability will be computed using addition
    for idx in range(0, vs):
        for s in range(0, 5):
            probs_multi[s, idx] = math.log(float(train_freqs[s, idx] + 1) / (vs + train_lens[s]))

        freq_n = train_freqs[0, idx] + train_freqs[1, idx] + train_freqs[2, idx]  # Negative frequencies
        freq_p = train_freqs[3, idx] + train_freqs[4, idx]  # Positive frequencies

        probs_bin[0, idx] = math.log(float(freq_n + 1) / (vs + len_n))
        probs_bin[1, idx] = math.log(float(freq_p + 1) / (vs + len_p))

    print("Done probabilities")

    predict_labels_multi = []
    predict_labels_bin = []

    tests = scipy.sparse.load_npz(to_system_path("{0}/freqs_doc-{1}.npz".format(data_dir, fold)))
    for i in range(0, tests.shape[0]):
        vec = tests[i].toarray()[0]
        test_probs = [0, 0, 0, 0, 0]
        for s in range(0, 5):
            test_probs[s] = numpy.dot(probs_multi[s], vec)  # Dot product, sum of prob * freq
        predict_labels_multi.append(find_label(test_probs) + 1)

        test_n = numpy.dot(probs_bin[0], vec)  # Dot product, sum of log(prob) * freq
        test_p = numpy.dot(probs_bin[1], vec)  # Dot product, sum of log(prob) * freq
        if test_n < test_p:
            predict_labels_bin.append("+")
        else:
            predict_labels_bin.append("-")

    actual_labels_multi = labels[test_fold-1]

    outf = open(to_system_path("{0}/labels-{1}.tsv".format(output_dir, test_fold)), "w")

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(0, len(predict_labels_multi)):
        actual_label = actual_labels_multi[i]
        predict_label_bin = predict_labels_bin[i]
        if actual_label > 3:
            outf.write("{0}\t{1}\t+\t{2}\n".format(actual_label, predict_labels_multi[i], predict_label_bin))
            if predict_label_bin == "+":
                tp += 1
            else:
                fn += 1
        else:
            outf.write("{0}\t{1}\t-\t{2}\n".format(actual_label, predict_labels_multi[i], predict_label_bin))
            if predict_label_bin == "+":
                fp += 1
            else:
                tn += 1

    outf.close()

    tps.append(tp)
    tns.append(tn)
    fps.append(fp)
    fns.append(fn)

    p, r, a, f, s = get_precision_recall_accuracy_f1_spc(actual_labels_multi, predict_labels_multi)
    precisions.append(p)
    recalls.append(r)
    accuracies.append(a)
    f1s.append(f)
    spcs.append(s)

    del tests, train_freqs, train_lens, probs_multi, probs_bin,
    del predict_labels_multi, predict_labels_bin
    print("Done fold {0}".format(test_fold))


def get_precision(fold):
    """ Compute precision for the give fold """
    return float(tps[fold]) / (tps[fold] + fps[fold])


def get_recall(fold):
    """ Compute recall for the give fold """
    return float(tps[fold]) / (tps[fold] + fns[fold])


def get_specificity(fold):
    """ Compute specificity for the give fold """
    return float(tns[fold]) / (tns[fold] + fps[fold])


def get_accuracy(fold):
    """ Compute accuracy for the give fold """
    return float(tps[fold] + tns[fold]) / (tps[fold] + tns[fold] + fps[fold] + fns[fold])


def get_f1(p, r):
    """ Compute F1 score for the give fold """
    return 2. * p * r / (p + r)


def to_percentage(r):
    """ Convert a float number of percentage with 2 decimals """
    return "{:.2%}".format(r)


max_num = max(max(tps), max(tns), max(fps), max(fns))
max_len = len(str(max_num))


def to_fixed_str(n):
    """ Force converting all integers to a fixed length string (prepending spaces) """
    s = str(n)
    while len(s) < max_len:
        s = " {0}".format(s)
    return s


print("Statistics for binary version")

# Save result for binary version
outf = open(to_system_path("{0}/result_binary.txt".format(output_dir)), "w")
outf.write("TP:")
for i in range(0, num_folds):
    outf.write("  {0}".format(to_fixed_str(tps[i])))
outf.write("\nTN:")
for i in range(0, num_folds):
    outf.write("  {0}".format(to_fixed_str(tns[i])))
outf.write("\nFP:")
for i in range(0, num_folds):
    outf.write("  {0}".format(to_fixed_str(fps[i])))
outf.write("\nFN:")
for i in range(0, num_folds):
    outf.write("  {0}".format(to_fixed_str(fns[i])))
outf.write("\n\nPrecision:")
ps = []
for i in range(0, num_folds):
    t = get_precision(i)
    ps.append(t)
    outf.write("  {0}".format(to_percentage(t)))
outf.write("    {0}\nRecall:   ".format(to_percentage(sum(ps) / num_folds)))
rs = []
for i in range(0, num_folds):
    t = get_recall(i)
    rs.append(t)
    outf.write("  {0}".format(to_percentage(t)))
outf.write("    {0}\nSPC:      ".format(to_percentage(sum(rs) / num_folds)))
ss = []
for i in range(0, num_folds):
    t = get_specificity(i)
    ss.append(t)
    outf.write("  {0}".format(to_percentage(t)))
outf.write("    {0}\nAccuracy: ".format(to_percentage(sum(ss) / num_folds)))
cs = []
for i in range(0, num_folds):
    t = get_accuracy(i)
    cs.append(t)
    outf.write("  {0}".format(to_percentage(t)))
outf.write("    {0}\nF-1 Score:".format(to_percentage(sum(cs) / num_folds)))
ts = 0.
for i in range(0, num_folds):
    f1 = get_f1(ps[i], rs[i])
    ts += f1
    outf.write("  {0}".format(to_percentage(f1)))
outf.write("    {0}\n".format(to_percentage(ts / num_folds)))
outf.close()

print("Statistics for multiclass version")

# Save result for multiclass version
outf = open(to_system_path("{0}/result_multiclass.txt".format(output_dir)), "w")
outf.write("Precision:")
for i in range(0, num_folds):
    outf.write("  {0}".format(to_percentage(precisions[i])))
outf.write("    {0}\nRecall:   ".format(to_percentage(sum(precisions) / num_folds)))
for i in range(0, num_folds):
    outf.write("  {0}".format(to_percentage(recalls[i])))
outf.write("    {0}\nSPC:      ".format(to_percentage(sum(recalls) / num_folds)))
for i in range(0, num_folds):
    outf.write("  {0}".format(to_percentage(spcs[i])))
outf.write("    {0}\nAccuracy: ".format(to_percentage(sum(spcs) / num_folds)))
for i in range(0, num_folds):
    outf.write("  {0}".format(to_percentage(accuracies[i])))
outf.write("    {0}\nF-1 Score:".format(to_percentage(sum(accuracies) / num_folds)))
for i in range(0, num_folds):
    outf.write("  {0}".format(to_percentage(f1s[i])))
outf.write("    {0}\n".format(to_percentage(sum(f1s) / num_folds)))
outf.close()

print("Done")
