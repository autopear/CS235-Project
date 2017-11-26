# Convert all reviews in all folds to tf-idf vectors
# Term Frequency: 1 + log(Ft), where Ft is the raw word frequency across all reviews
# Inverse Document Frequency: log(1 + N / Nt), where N is the total number of documents, Nt is the number of documents
#   that contain word t

import glob
import gzip
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


dir_path = to_standard_path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # Module directory
fold_prefix = to_system_path("{0}/fastText/data/fold-".format(os.path.dirname(dir_path)))
words_path = to_system_path("{0}/Samples/word_map.tsv".format(dir_path))
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

doc_freqs = {}  # Document frequencies (number of documents that contain a word)
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
            # Use set() to remove duplicates, so each word in counted once per document
            words = set(components[2].split(" "))
            for word in words:
                idx = words_map[word]  # Use word index instead of text to be later used as column index
                doc_freqs[idx] = doc_freqs.get(idx, 0) + 1
    inf.close()
    if fold_size == 0:
        fold_size = num_docs  # Read the number of documents per fold dynamically
    print("Done {0}".format(os.path.basename(fold_path)))

# Convert doc_freqs to IDF
for word in sorted(doc_freqs.keys()):
    doc_freqs[word] = math.log(1 + float(num_docs) / doc_freqs[word])
print("Done inverse document frequencies")

for fold_path in glob.glob(to_system_path("{0}*.tsv.gzip").format(fold_prefix)):
    fold = int(os.path.basename(fold_path)[5:-9])
    label_path = "{0}/labels-{1}.txt".format(data_dir, fold)
    labelf = open(label_path, "w")

    mtx = numpy.zeros((fold_size, num_words), dtype=float)  # Create a 2D array for documents and tf-idf

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
            word_freqs = {}  # Raw frequencies for words in one review
            for word in words:
                idx = words_map[word]  # Use word index instead of text to be later used as column index
                word_freqs[idx] = word_freqs.get(idx, 0) + 1
            # Compute tf-idf for each word
            for idx, rtf in word_freqs.items():
                tf = 1 + math.log(rtf)  # Log normalized TF
                idf = doc_freqs[idx]
                tfidf = tf * idf
                # line_num (start from 0) is row index for the document
                # idx (stars from 1) is the column index for the word
                mtx[line_num, idx-1] = tfidf
            labelf.write("{0}\t{1}\n".format(doc_idx, score))
            line_num += 1
    inf.close()
    labelf.close()

    # Convert the 2D tf-idf array to sparse matrix, and save to a compressed file
    out_path = "{0}/matrix-{1}.npz".format(data_dir, fold)
    scipy.sparse.save_npz(out_path, csr_matrix(mtx), compressed=True)
    print("Done fold {0}".format(fold))

print("Done")
