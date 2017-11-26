# fastText accepts direct words, so revert 5 groups back to word instead of word indices

import gzip
import os

num_folds = 5


def to_system_path(path):
    """ Convert an input path to the current system style, \ for Windows, / for others """
    if os.name == "nt":
        return path.replace("/", "\\")
    else:
        return path.replace("\\", "/")


def to_standard_path(path):
    """ Convert \ to / in path (mainly for Windows) """
    return path.replace("\\", "/")


dir_path = to_standard_path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
output_dir = "{0}/data".format(dir_path)
input_dir = "{0}/Samples".format("/".join(dir_path.split("/")[:-1]))
word_map_path = to_system_path("{0}/word_map.tsv".format(input_dir))
fold_prefix = to_system_path("{0}/5-fold/fold-".format(input_dir))

word_map = {}
with open(word_map_path, "r") as inf:
    for line in inf:
        line = line[:-1]
        if len(line) < 1:
            continue
        components = line.split("\t")
        idx = int(components[0])
        word = components[1]
        word_map[idx] = word
inf.close()
print("Word map loaded")

for i in range(1, num_folds+1):
    fold_path = "{0}{1}.tsv.gzip".format(fold_prefix, i)

    outf = gzip.open(to_system_path("{0}/fold-{1}.tsv.gzip".format(output_dir, i)), "wt")

    with gzip.open(fold_path, "rt") as inf:
        for line in inf:
            line = line[:-1]
            if len(line) < 1:
                continue
            components = line.split("\t")
            doc_idx = int(components[0])
            score = int(components[1])
            word_indices = components[2].split(" ")
            words = []
            for idx in word_indices:
                words.append(word_map[int(idx)])
            outf.write("{0}\t{1}\t{2}\n".format(doc_idx, score, " ".join(words)))
    inf.close()
    outf.close()
    print("Done fold {0}".format(i))
print("Done")
