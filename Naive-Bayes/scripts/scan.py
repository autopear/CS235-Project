# Scan for word frequencies and review lengths


import glob
import gzip
import numpy
import os
import scipy
from scipy.sparse import csr_matrix


num_reviews = 40000  # Number of reviews per fold


def to_system_path(path):
    """ Convert an input path to the current system style, \ for Windows, / for others """
    if os.name == "nt":
        return path.replace("/", "\\")
    else:
        return path.replace("\\", "/")


def to_standard_path(path):
    """ Convert \ to / in path (mainly for Windows) """
    return path.replace("\\", "/")


dir_path = to_standard_path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # Module directory
src_dir = to_standard_path("{0}/Samples/5-fold".format(os.path.dirname(dir_path)))
words_path = to_system_path("{0}/Samples/word_map.tsv".format(os.path.dirname(dir_path)))
data_dir = "{0}/data".format(dir_path)
output_dir = "{0}/output".format(dir_path)

# Read number of folds dynamically
folds = []
for fp in glob.glob(to_system_path("{0}/fold-*.tsv.gzip".format(src_dir))):
    folds.append(int(os.path.basename(fp)[5:-9]))
folds.sort()

num_words = 0
with open(words_path, "r") as inf:
    line = inf.readlines()[-1]  # Read the last line's first column
    num_words = int(line.split("\t")[0])
inf.close()
print("{0} words loaded".format(num_words))

for fold in folds:
    freqs_doc = numpy.zeros((num_reviews, num_words), dtype=int)  # Word frequency per review
    freqs_class = numpy.zeros((5, num_words), dtype=int)  # Word frequency per class

    outf = open(to_system_path("{0}/fold-{1}.tsv".format(data_dir, fold)), "w")
    with gzip.open(to_system_path("{0}/fold-{1}.tsv.gzip".format(src_dir, fold)), "rt") as inf:
        line_num = 0
        for line in inf:
            line = line[:-1]
            components = line.split("\t")
            if len(components) != 3:
                continue

            score = int(components[1])
            words = components[2].split(" ")
            outf.write("{0}\t{1}\n".format(score, len(words)))  # Save review's label and length
            cnts = {}  # Use dict for scan to reduce the number of updates in the matrix
            for word in words:
                idx = int(word) - 1
                cnts[idx] = cnts.get(idx, 0) + 1
            for idx, cnt in cnts.items():
                freqs_doc[line_num, idx] = cnt
                freqs_class[score-1, idx] = freqs_class[score-1, idx] + cnt
            line_num += 1
    inf.close()
    outf.close()

    # Save frequency matrix (convert to sparse matrix first) to file
    out_path = to_system_path("{0}/freqs_doc-{1}.npz".format(data_dir, fold))
    scipy.sparse.save_npz(out_path, csr_matrix(freqs_doc), compressed=True)

    out_path = to_system_path("{0}/freqs_class-{1}.npz".format(data_dir, fold))
    scipy.sparse.save_npz(out_path, csr_matrix(freqs_class), compressed=True)

    print("Done fold {0}".format(fold))

print("Done")
