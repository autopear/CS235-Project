import numpy
import os
import glob
import scipy
import math
from scipy.sparse import csr_matrix, vstack
from pysparnn.cluster_index import ClusterIndex
from operator import itemgetter


fold_size = 40000
neighbors = 20


def to_system_path(path):
    if os.name == "nt":
        return path.replace("/", "\\")
    else:
        return path.replace("\\", "/")


def to_standard_path(path):
    return path.replace("\\", "/")


def write_labels(lines, path):
    outf = open(path, "w")
    outf.write("\n".join(lines))
    outf.write("\n")
    outf.close()


dir_path = to_standard_path(os.path.dirname(os.path.realpath(__file__)))
dir_path = "/".join(dir_path.split("/")[:-1])
words_path = to_system_path("{0}/Samples/word_map.tsv".format("/".join(dir_path.split("/")[:-1])))
data_dir = to_system_path("{0}/data".format(dir_path))
output_dir = to_system_path("{0}/output".format(dir_path))

bin_weighted_prefix = "{0}/binary_weighted-".format(output_dir)
bin_unweighted_prefix = "{0}/binary_unweighted-".format(output_dir)
multi_weighted_prefix = "{0}/multiclass_weighted-".format(output_dir)
multi_unweighted_prefix = "{0}/multiclass_unweighted-".format(output_dir)

if neighbors % 2 == 0:
    neighbors += 1  # Make it odd

num_words = 0
with open(words_path, "r") as inf:
    line = inf.readlines()[-1]
    num_words = int(line.split("\t")[0])
inf.close()
print("{0} words loaded".format(num_words))

num_folds = len(glob.glob(to_system_path("{0}/matrix-*.npz".format(data_dir))))

folds = []
matrices = [None] * num_folds
docs = numpy.zeros((num_folds, fold_size), dtype=int)
labels = {}
for fold_path in glob.glob(to_system_path("{0}/matrix-*.npz".format(data_dir))):
    fold = int(os.path.basename(fold_path)[7:-4])

    matrices[fold-1] = scipy.sparse.load_npz(to_system_path("{0}/matrix-{1}.npz".format(data_dir, fold)))

    with open(to_system_path("{0}/labels-{1}.txt".format(data_dir, fold)), "r") as inf:
        line_num = 0
        for line in inf:
            line = line[:-1]
            components = line.split("\t")
            if len(components) == 2:
                doc_idx = int(components[0])
                score = int(components[1])
                labels[doc_idx] = score
                docs[fold-1, line_num] = doc_idx
            line_num += 1
    inf.close()

    folds.append(fold)
folds.sort()
print("Matrices and labels loaded")

for test_fold in folds:
    space = None
    train_docs = []

    for train_fold in folds:
        if train_fold == test_fold:
            continue

        for i in range(0, fold_size):
            train_docs.append(docs[train_fold - 1, i])

        train_vecs = matrices[train_fold-1]
        if space is None:
            space = train_vecs
        else:
            space = vstack([space, train_vecs])

    print("Set fold {0} for testing".format(test_fold))

    clusters = ClusterIndex(space, train_docs)
    print("Done clusters")

    nns = clusters.search(matrices[test_fold - 1], k=neighbors, k_clusters=3, return_distance=True)
    del space, clusters, train_docs


    lines_bin_unweighted = []
    lines_bin_weighted = []
    lines_multi_unweighted = []
    lines_multi_weighted = []

    for i in range(0, fold_size):
        knns = nns[i]
        actual_label = labels[docs[test_fold-1, i]]

        w_bin = 0
        w_multi = 0
        scores_bin = {}
        scores_multi = {}

        for cosd, doc_idx in knns:
            pl = labels[doc_idx]
            w_multi += (1-cosd) * pl
            if pl > 3:
                w_bin += (1 - cosd)
            else:
                w_bin += (cosd - 1)

            scores_multi[pl] = scores_multi.get(pl, 0) + 1
            if pl > 3:
                scores_bin["+"] = scores_bin.get("+", 0) + 1
            else:
                scores_bin["-"] = scores_bin.get("-", 0) + 1

        w_multi = w_multi / len(knns)

        if w_multi > 0:
            if actual_label > 3:
                lines_bin_weighted.append("+\t+")
            else:
                lines_bin_weighted.append("-\t+")
        else:
            if actual_label > 3:
                lines_bin_weighted.append("+\t-")
            else:
                lines_bin_weighted.append("-\t-")

        predict_label_multi = int(max(scores_multi.items(), key=itemgetter(1))[0])
        predict_label_bin = max(scores_bin.items(), key=itemgetter(1))[0]
        del scores_multi, scores_bin

        lines_multi_weighted.append("{0}\t{1}".format(actual_label, int(math.ceil(w_multi))))
        lines_multi_unweighted.append("{0}\t{1}".format(actual_label, predict_label_multi))

        if actual_label > 3:
            if w_bin > 0:
                lines_bin_weighted.append("+\t+")
            else:
                lines_bin_weighted.append("+\t-")
            lines_bin_unweighted.append("+\t{0}".format(predict_label_bin))
        else:
            if w_bin > 0:
                lines_bin_weighted.append("-\t+")
            else:
                lines_bin_weighted.append("-\t-")
            lines_bin_unweighted.append("-\t{0}".format(predict_label_bin))
    del nns

    write_labels(lines_bin_unweighted, "{0}{1}.tsv".format(bin_unweighted_prefix, test_fold))
    write_labels(lines_bin_weighted, "{0}{1}.tsv".format(bin_weighted_prefix, test_fold))
    write_labels(lines_multi_unweighted, "{0}{1}.tsv".format(multi_unweighted_prefix, test_fold))
    write_labels(lines_multi_weighted, "{0}{1}.tsv".format(multi_weighted_prefix, test_fold))
    del lines_multi_weighted, lines_multi_unweighted, lines_bin_unweighted, lines_bin_weighted

    print("Done fold {0}".format(test_fold))

print("Done")