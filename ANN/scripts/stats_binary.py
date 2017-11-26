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
label_prefixes = (to_system_path("{0}/binary_unweighted-".format(output_dir)), to_system_path("{0}/binary_weighted-".format(output_dir)))
output_path = to_system_path("{0}/result_binary.txt".format(output_dir))

folds = []
for fold_path in glob.glob("{0}*.tsv".format(label_prefixes[0])):
    folds.append(int(os.path.basename(fold_path)[18:-4]))
folds.sort()
num_folds = len(folds)

contents = []

for label_prefix in label_prefixes:
    tps = []
    tns = []
    fps = []
    fns = []

    for predict_fold in folds:
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        with open("{0}{1}.tsv".format(label_prefix, predict_fold), "r") as inf:
            for line in inf:
                line = line[:-1]
                if len(line) < 1:
                    continue
                components = line.split("\t")
                actual_label = components[0]
                predict_label = components[1]
                if (actual_label == "+") and (predict_label == "+"):
                    tp += 1
                elif (actual_label == "-") and (predict_label == "+"):
                    fp += 1
                elif (actual_label == "+") and (predict_label == "-"):
                    fn += 1
                else:
                    tn += 1
        inf.close()

        tps.append(tp)
        tns.append(tn)
        fps.append(fp)
        fns.append(fn)


    def get_precision(fold):
        return float(tps[fold]) / (tps[fold] + fps[fold])


    def get_recall(fold):
        return float(tps[fold]) / (tps[fold] + fns[fold])


    def get_specificity(fold):
        return float(tns[fold]) / (tns[fold] + fps[fold])


    def get_accuracy(fold):
        return float(tps[fold] + tns[fold]) / (tps[fold] + tns[fold] + fps[fold] + fns[fold])


    def get_f1(p, r):
        return 2. * p * r / (p + r)


    def to_percentage(r):
        return "{:.2%}".format(r)


    max_num = max(max(tps), max(tns), max(fps), max(fns))
    max_len = len(str(max_num))


    def to_fixed_str(n):
        s = str(n)
        while len(s) < max_len:
            s = " {0}".format(s)
        return s

    lines = []
    
    lines.append("TP:")
    for i in range(0, num_folds):
        lines.append("  {0}".format(to_fixed_str(tps[i])))
    lines.append("\nTN:")
    for i in range(0, num_folds):
        lines.append("  {0}".format(to_fixed_str(tns[i])))
    lines.append("\nFP:")
    for i in range(0, num_folds):
        lines.append("  {0}".format(to_fixed_str(fps[i])))
    lines.append("\nFN:")
    for i in range(0, num_folds):
        lines.append("  {0}".format(to_fixed_str(fns[i])))
    lines.append("\n\nPrecision:")
    ps = []
    for i in range(0, num_folds):
        t = get_precision(i)
        ps.append(t)
        lines.append("  {0}".format(to_percentage(t)))
    lines.append("    {0}\nRecall:   ".format(to_percentage(sum(ps) / num_folds)))
    rs = []
    for i in range(0, num_folds):
        t = get_recall(i)
        rs.append(t)
        lines.append("  {0}".format(to_percentage(t)))
    lines.append("    {0}\nSPC:      ".format(to_percentage(sum(rs) / num_folds)))
    ss = []
    for i in range(0, num_folds):
        t = get_specificity(i)
        ss.append(t)
        lines.append("  {0}".format(to_percentage(t)))
    lines.append("    {0}\nAccuracy: ".format(to_percentage(sum(ss) / num_folds)))
    cs = []
    for i in range(0, num_folds):
        t = get_accuracy(i)
        cs.append(t)
        lines.append("  {0}".format(to_percentage(t)))
    lines.append("    {0}\nF-1 Score:".format(to_percentage(sum(cs) / num_folds)))
    ts = 0.
    for i in range(0, num_folds):
        f1 = get_f1(ps[i], rs[i])
        ts += f1
        lines.append("  {0}".format(to_percentage(f1)))
    lines.append("    {0}\n".format(to_percentage(ts / num_folds)))

    contents.append("".join(lines))

# Save result
outf = open(output_path, "w")
outf.write("Unweighted\n")
outf.write(contents[0])
outf.write("\n\nWeighted\n")
outf.write(contents[1])
outf.close()

print("Done")
