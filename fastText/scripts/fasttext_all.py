# Merge 4 folds and build fastText classifier from them. Then test against the remaining 1 fold.

import os
import gzip
import shutil

num_folds = 5
num_threads = 4


def to_system_path(path):
    if os.name == "nt":
        return path.replace("/", "\\")
    else:
        return path.replace("\\", "/")


def to_standard_path(path):
    return path.replace("\\", "/")


dir_path = to_standard_path(os.path.dirname(os.path.realpath(__file__)))
bin_path = ""
if os.name == 'nt':
    bin_path = to_system_path("{0}/fasttext/fasttext.exe".format(dir_path))
else:
    bin_path = to_system_path("{0}/fasttext/fasttext".format(dir_path))

# Download fastText' source and compile for the binary
if not os.path.isfile(bin_path):
    if not os.path.isdir(to_system_path("{0}/fasttext".format(dir_path))):
        os.system("cd \"{0}\" && git clone https://github.com/facebookresearch/fastText.git".format(dir_path))
    os.system("cd \"{0}\" && cd fasttext && make".format(dir_path))

dir_path = "/".join(dir_path.split("/")[:-1])
data_dir = "{0}/data".format(dir_path)
output_dir = "{0}/output".format(dir_path)
tmp_dir = "{0}/tmp".format(dir_path)
fold_prefix = to_system_path("{0}/fold-".format(data_dir))
model_path = to_system_path("{0}/model".format(tmp_dir))

# Create temporary work space
if not os.path.isdir(to_system_path(tmp_dir)):
    os.makedirs(to_system_path(tmp_dir))

train_path = to_system_path("{0}/train.csv".format(tmp_dir))
test_path = to_system_path("{0}/test.txt".format(tmp_dir))
model_path = to_system_path("{0}/model".format(tmp_dir))
predict_path = to_system_path("{0}/labels".format(tmp_dir))


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


precisions = []
recalls = []
accuracies = []
f1s = []
spcs = []

for test_fold in range(1, 1 + num_folds):
    print("Set fold {0} for testing".format(test_fold))

    train_file = open(train_path, "w")

    for i in range(1, num_folds + 1):
        if i == test_fold:
            continue
        with gzip.open("{0}{1}.tsv.gzip".format(fold_prefix, i), "rt") as inf:
            for line in inf:
                line = line[:-1]
                if len(line) < 1:
                    continue
                components = line.split("\t")
                score = int(components[1])
                words = components[2]
                train_file.write("__label__{0} {1}\n".format(score, words))
        inf.close()
    train_file.close()
    print("Done training file")

    test_file = open(test_path, "w")
    test_labels = []
    with gzip.open("{0}{1}.tsv.gzip".format(fold_prefix, test_fold), "rt") as inf:
        for line in inf:
            line = line[:-1]
            if len(line) < 1:
                continue
            components = line.split("\t")
            test_labels.append(int(components[1]))
            words = components[2]
            test_file.write("{0}\n".format(words))
    inf.close()
    test_file.close()
    print("Done testing file")

    # Learn model
    os.system("\"{0}\" supervised -input \"{1}\" -output \"{2}\" -thread {3}".format(bin_path, train_path, model_path, num_threads))
    print("Done model")

    # Prediction for test set
    os.system("\"{0}\" predict \"{1}.bin\" \"{2}\" > \"{3}\"".format(bin_path, model_path, test_path, predict_path))
    print("Done prediction")

    predict_labels = []
    with open(predict_path, "r") as inf:
        for line in inf:
            line = line[:-1]
            if len(line) < 1:
                continue
            predict_labels.append(int(line.replace("__label__", "")))
    inf.close()

    labels_path = to_system_path("{0}/labels_all-{1}.tsv".format(output_dir, test_fold))
    outf = open(labels_path, "w")
    for i in range(0, len(test_labels)):
        outf.write("{0}\t{1}\n".format(test_labels[i], predict_labels[i]))
    outf.close()

    p, r, a, f, s = get_precision_recall_accuracy_f1_spc(test_labels, predict_labels)
    precisions.append(p)
    recalls.append(r)
    accuracies.append(a)
    f1s.append(f)
    spcs.append(s)

    del test_labels, predict_labels

    print("Done testing fold {0}".format(test_fold))

# Delete tmp
shutil.rmtree(to_system_path(tmp_dir))


def to_percentage(r):
    return "{:.2%}".format(r)


# Save result
outf = open(to_system_path("{0}/result_all.txt".format(output_dir)), "w")
outf.write("Precision:")
ps = []
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
