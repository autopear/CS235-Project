# Merge 4 folds and build fastText classifier from them. Then test against the remaining 1 fold.
# This is a binary version of reviews. Score 4 and 5 are considered positive, 1 to 3 are negative.

import os
import gzip
import glob
import math
import numpy
import scipy
from scipy.sparse import csr_matrix


N = 2


def to_system_path(path):
    if os.name == "nt":
        return path.replace("/", "\\")
    else:
        return path.replace("\\", "/")


def to_standard_path(path):
    return path.replace("\\", "/")


dir_path = to_standard_path(os.path.dirname(os.path.realpath(__file__)))
fold_prefix = to_system_path("{0}/fastText/data/fold-".format("/".join(dir_path.split("/")[:-2])))
dir_path = "/".join(dir_path.split("/")[:-1])
words_path = to_system_path("{0}/Samples/word_map.tsv".format("/".join(dir_path.split("/")[:-1])))
data_dir = to_system_path("{0}/data".format(dir_path))

words_map = {}
with open(words_path, "r") as inf:
    for line in inf:
        line = line[:-1]
        components = line.split("\t")
        if len(components) == 2:
            idx = int(components[0])
            word = components[1]
            words_map[word] = idx
inf.close()
num_words = len(words_map)
print("Done word list")

doc_freqs = {}
num_docs = 0

fold_size = 0

for fold_path in glob.glob(to_system_path("{0}*.tsv.gzip").format(fold_prefix)):
    with gzip.open(fold_path, "rt") as inf:
        for line in inf:
            line = line[:-1]
            components = line.split("\t")
            if len(components) != 3:
                continue
            num_docs += 1
            words = components[2].split(" ")
            for word in set(words):
                idx = words_map[word]
                doc_freqs[idx] = doc_freqs.get(idx, 0) + 1
    inf.close()
    if fold_size == 0:
        fold_size = num_docs
    print("Done {0}".format(os.path.basename(fold_path)))

for word in sorted(doc_freqs.keys()):
    doc_freqs[word] = math.log(1 + float(num_docs) / doc_freqs[word])
print("Done document frequencies")

for fold_path in glob.glob(to_system_path("{0}*.tsv.gzip").format(fold_prefix)):
    fold = int(os.path.basename(fold_path)[5:-9])
    label_path = "{0}/labels-{1}.txt".format(data_dir, fold)
    labelf = open(label_path, "w")

    mtx = numpy.zeros((fold_size, num_words), dtype=float)

    with gzip.open(fold_path, "rt") as inf:
        line_num = 0
        for line in inf:
            line = line[:-1]
            components = line.split("\t")
            if len(components) != 3:
                continue
            doc_idx = int(components[0])
            score = int(components[1])
            words = components[2].split(" ")
            word_freqs = {}
            for word in words:
                idx = words_map[word]
                word_freqs[idx] = word_freqs.get(idx, 0) + 1
            t = []
            for idx, rtf in word_freqs.items():
                tf = 1 + math.log(rtf)
                idf = doc_freqs[idx]
                tfidf = tf * idf
                mtx[line_num, idx-1] = tfidf
            labelf.write("{0}\t{1}\n".format(doc_idx, score))
            line_num += 1
    inf.close()
    labelf.close()

    out_path = "{0}/matrix-{1}.npz".format(data_dir, fold)
    scipy.sparse.save_npz(out_path, csr_matrix(mtx), compressed=True)
    print("Done fold {0}".format(fold))

print("Done")
