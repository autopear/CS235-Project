# Merge 4 folds and build fastText classifier from them. Then test against the remaining 1 fold.
# This is a binary version of reviews. Score 4 and 5 are considered positive, 1 to 3 are negative.

import gzip
import os
import shutil

num_folds = 5
num_threads = 4


def to_system_path(path):
    """ Convert an input path to the current system style, \ for Windows, / for others """
    if os.name == "nt":
        return path.replace("/", "\\")
    else:
        return path.replace("\\", "/")


def to_standard_path(path):
    """ Convert \ to \ in path (mainly for Windows) """
    return path.replace("\\", "/")


dir_path = to_standard_path(os.path.dirname(os.path.realpath(__file__)))
bin_path = ""
if os.name == 'nt':
    bin_path = to_system_path("{0}/fasttext/fasttext.exe".format(dir_path))
else:
    bin_path = to_system_path("{0}/fasttext/fasttext".format(dir_path))

bin_found = True
# Download fastText' source and compile for the binary
if not os.path.isfile(bin_path):
    if os.name == "nt":
        print("Please build fastText binary")
        bin_found = False
    else:
        if not os.path.isdir(to_system_path("{0}/fasttext".format(dir_path))):
            os.system("cd \"{0}\" && git clone https://github.com/facebookresearch/fastText.git".format(dir_path))
        os.system("cd \"{0}\" && cd fasttext && make".format(dir_path))

if bin_found:
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

    tps = []  # True positive
    tns = []  # True negative
    fps = []  # False positive
    fns = []  # False negative

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
                    if score > 3:
                        train_file.write("__label__+ {0}\n".format(words))
                    else:
                        train_file.write("__label__- {0}\n".format(words))
            inf.close()
        train_file.close()
        print("Done training file")

        test_file = open(test_path, "w")
        actual_labels = []
        with gzip.open("{0}{1}.tsv.gzip".format(fold_prefix, test_fold), "rt") as inf:
            for line in inf:
                line = line[:-1]
                if len(line) < 1:
                    continue
                components = line.split("\t")
                score = int(components[1])
                if score > 3:
                    actual_labels.append("+")
                else:
                    actual_labels.append("-")
                words = components[2]
                test_file.write("{0}\n".format(words))
        inf.close()
        test_file.close()
        print("Done testing file")

        # Learn model
        os.system(
            "\"{0}\" supervised -input \"{1}\" -output \"{2}\" -thread {3}".format(bin_path, train_path, model_path,
                                                                                   num_threads))
        print("Done model")

        # Prediction for test set
        os.system("\"{0}\" predict \"{1}.bin\" \"{2}\" > \"{3}\"".format(bin_path, model_path, test_path, predict_path))
        print("Done prediction")

        # Read predicted labels
        predict_labels = []
        with open(predict_path, "r") as inf:
            for line in inf:
                line = line[:-1]
                if len(line) < 1:
                    continue
                predict_labels.append(line.replace("__label__", ""))
        inf.close()

        labels_path = to_system_path("{0}/labels_binary-{1}.tsv".format(output_dir, test_fold))
        outf = open(labels_path, "w")

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        # Write to output with corresponding real labels and predicted labels, together with computing some values
        for i in range(0, len(actual_labels)):
            test_label = actual_labels[i]
            predict_label = predict_labels[i]
            if (test_label == "+") and (predict_label == "+"):
                tp += 1
            elif (test_label == "-") and (predict_label == "+"):
                fp += 1
            elif (test_label == "+") and (predict_label == "-"):
                fn += 1
            else:
                tn += 1
            outf.write("{0}\t{1}\n".format(actual_labels[i], predict_labels[i]))
        outf.close()
        del actual_labels, predict_labels

        tps.append(tp)
        tns.append(tn)
        fps.append(fp)
        fns.append(fn)

        print("Done testing fold {0}".format(test_fold))

    # Delete tmp
    shutil.rmtree(to_system_path(tmp_dir))


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


    # Save result
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

    print("Done")
