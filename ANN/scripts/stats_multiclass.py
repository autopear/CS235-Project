import os
import gzip
import glob
import math
from operator import itemgetter


def to_system_path(path):
    if os.name == "nt":
        return path.replace("/", "\\")
    else:
        return path.replace("\\", "/")


def to_standard_path(path):
    return path.replace("\\", "/")


dir_path = to_standard_path(os.path.dirname(os.path.realpath(__file__)))
dir_path = "/".join(dir_path.split("/")[:-1])
output_dir = to_system_path("{0}/output".format(dir_path))
label_prefixes = (to_system_path("{0}/multiclass_unweighted-".format(output_dir)), to_system_path("{0}/multiclass_weighted-".format(output_dir)))
output_path = to_system_path("{0}/result_multiclass.txt".format(output_dir))

folds = []
for fold_path in glob.glob("{0}*.tsv".format(label_prefixes[0])):
    folds.append(int(os.path.basename(fold_path)[22:-4]))
folds.sort()
num_folds = len(folds)


def get_binary_precision_recall_f1_spc(reals, preds, true_value):
    # Compute precision, recall and f1-score for setting true_value as binary positive
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
    # Compute precision, recall, accuracy and f1-score for all classes
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


contents = []

for label_prefix in label_prefixes:

    precisions = []
    recalls = []
    accuracies = []
    f1s = []
    spcs = []

    for predict_fold in folds:
        actual_labels = []
        predict_labels = []
        with open("{0}{1}.tsv".format(label_prefix, predict_fold), "r") as inf:
            for line in inf:
                line = line[:-1]
                if len(line) < 1:
                    continue
                components = line.split("\t")
                actual_labels.append(int(components[0]))
                predict_labels.append(int(components[1]))
        inf.close()

        p, r, a, f, s = get_precision_recall_accuracy_f1_spc(actual_labels, predict_labels)
        precisions.append(p)
        recalls.append(r)
        accuracies.append(a)
        f1s.append(f)
        spcs.append(s)

        del actual_labels, predict_labels


    def to_percentage(r):
        return "{:.2%}".format(r)

    lines = []

    # Save result
    lines.append("Precision:")
    ps = []
    for i in range(0, num_folds):
        lines.append("  {0}".format(to_percentage(precisions[i])))
    lines.append("    {0}\nRecall:   ".format(to_percentage(sum(precisions) / num_folds)))
    for i in range(0, num_folds):
        lines.append("  {0}".format(to_percentage(recalls[i])))
    lines.append("    {0}\nSPC:      ".format(to_percentage(sum(recalls) / num_folds)))
    for i in range(0, num_folds):
        lines.append("  {0}".format(to_percentage(spcs[i])))
    lines.append("    {0}\nAccuracy: ".format(to_percentage(sum(spcs) / num_folds)))
    for i in range(0, num_folds):
        lines.append("  {0}".format(to_percentage(accuracies[i])))
    lines.append("    {0}\nF-1 Score:".format(to_percentage(sum(accuracies) / num_folds)))
    for i in range(0, num_folds):
        lines.append("  {0}".format(to_percentage(f1s[i])))
    lines.append("    {0}\n".format(to_percentage(sum(f1s) / num_folds)))

    contents.append("".join(lines))

outf = open(output_path, "w")
outf.write("Unweighted\n")
outf.write(contents[0])
outf.write("\n\nWeighted\n")
outf.write(contents[1])
outf.close()

print("Done")
