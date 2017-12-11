import os
import random
import linecache
import numpy as np
import scipy.sparse
import itertools

from scipy.sparse import dok_matrix, csr_matrix, coo_matrix

# Expects tsv not gzip
# Samples without replacement from folds

def load_word_map():
    word_map = {}
    path = os.path.join("..", "Samples", "word_map.tsv")
    with open(path, "r") as file:
        for line in file:
            (key, val) = line.split()
            key = int(key)
            word_map[key] = val
    return word_map


# Returns tf of a fold and respective labels
def get_fold_tf(fold_num, num_words, sample_indices):
    folder_path = os.path.join("..", "Samples", "5-fold")
    path = os.path.join(folder_path, "fold-" + str(fold_num) + ".tsv")

    doc_idx = 0
    term_freq = dok_matrix((len(sample_indices), num_words))
    labels = []
    for i in sample_indices:
        line = linecache.getline(path, i)
        doc_features = line.split()
        doc_length = len(doc_features[2:])

        label = -1
        if int(doc_features[1]) >= 4:
            label = 1

        labels.append(label)

        for word_idx in doc_features[2:]:
            term_freq[doc_idx, int(word_idx) - 1] += 1

        term_freq[doc_idx, :] /= doc_length
        doc_idx += 1

    term_freq = coo_matrix.log1p(coo_matrix(term_freq))

    labels = np.asarray(labels)
    labels = np.reshape(labels, (-1, 1))

    return term_freq, labels


# Returns idf of expected set of folds
def get_fold_idf(fold_nums, num_words, sample_indices):
    num_total_doc = len(sample_indices) * len(fold_nums)

    id_freq = np.ones((1, num_words))

    folder_path = os.path.join("..", "Samples", "5-fold")
    for fold_num in fold_nums:
        if 1 <= fold_num <= 5:
            path = os.path.join(folder_path, "fold-" + str(fold_num) + ".tsv")
            for i in sample_indices:
                line = linecache.getline(path, i)
                doc_features = line.split()

                # Don't want to double count a word we have already seen in a document
                seen_words = set()
                for word_idx in doc_features[2:]:
                    if int(word_idx) - 1 not in seen_words:
                        id_freq[0, int(word_idx) - 1] += 1
                        seen_words.add(int(word_idx) - 1)

    id_freq = coo_matrix(np.log(1 + (num_total_doc / id_freq)))

    return id_freq


if __name__ == "__main__":
    tf_folder_path = os.path.join("tf-idf_data", "tf")
    idf_folder_path = os.path.join("tf-idf_data", "idf")
    label_folder_path = os.path.join("tf-idf_data", "labels")

    if not os.path.exists(tf_folder_path):
        os.makedirs(tf_folder_path)
    if not os.path.exists(idf_folder_path):
        os.makedirs(idf_folder_path)
    if not os.path.exists(label_folder_path):
        os.makedirs(label_folder_path)

    word_map = load_word_map()

    folds = [1, 2, 3, 4, 5]

    # Indices of file to sample from
    indices = random.sample(range(0, 40000-1), 1250)

    for fold in folds:
        tf_path = os.path.join(tf_folder_path, "tf_fold_" + str(fold))
        label_path = os.path.join(label_folder_path, "label_fold_" + str(fold))

        fold_tf, fold_labels = get_fold_tf(fold, num_words=len(word_map), sample_indices=indices)

        scipy.sparse.save_npz(tf_path, fold_tf)
        np.save(label_path, fold_labels)

        print("Saved TF " + str(tf_path))
        print("Saved label " + str(label_path))

    five_fold = set(folds)
    fold_combinations = itertools.combinations(five_fold, len(five_fold)-1)
    for combination in fold_combinations:
        idf_path = os.path.join(idf_folder_path, "idf_fold_" + "_".join(map(str, combination)))
        fold_idf = get_fold_idf(combination, num_words=len(word_map), sample_indices=indices)
        scipy.sparse.save_npz(idf_path, fold_idf)
        print("Saved IDF " + str(idf_path))