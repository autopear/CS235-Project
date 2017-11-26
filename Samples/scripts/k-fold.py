# Randomly split all reviews in to k groups. So we can use k-1 groups for training, and 1 group for testing.
# People usually use 5 groups, which is 5-fold cross validation.


import glob
import gzip
import os
import random
from operator import itemgetter


k = 5  # Number of folds


def to_system_path(path):
    """ Convert an input path to the current system style, \ for Windows, / for others """
    if os.name == "nt":
        return path.replace("/", "\\")
    else:
        return path.replace("\\", "/")


def to_standard_path(path):
    """ Convert \ to / in path (mainly for Windows) """
    return path.replace("\\", "/")


dir_path = to_standard_path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # Module folder
input_paths = []
output_prefix = to_system_path("{0}/5-fold/fold-".format(dir_path))

# Scan input files
if os.path.isfile(to_system_path("{0}/samples_indices.tsv.gzip".format(dir_path))):
    input_paths.append(to_system_path("{0}/samples_indices.tsv.gzip".format(dir_path)))
else:
    for f in glob.glob(to_system_path("{0}/samples_indices-*.tsv.gzip".format(dir_path))):
        input_paths.append(to_system_path(f))

docs = []  # A list for all sampled reviews
for input_path in input_paths:
    with gzip.open(input_path, "rt") as inf:
        for line in inf:
            line = line[:-1]
            if len(line) < 1:
                continue
            components = line.split("\t")
            doc_idx = int(components[0])  # Document ID
            score = int(components[1])  # Rating
            doc = components[2]  # Review text
            docs.append((doc_idx, score, doc))
    inf.close()
    print("Loaded {0}".format(input_path))
print("{0} documents loaded".format(len(docs)))

# Shuffle 3 times to make reviews randomly distributed
random.shuffle(docs)
random.shuffle(docs)
random.shuffle(docs)

fold_size = int(len(docs) / k)  # Number of reviews per fold

for i in range(1, k+1):
    # Extract fold_size reviews and remove them from the all reviews list
    fold_docs = docs[0:fold_size]
    del docs[0:fold_size]

    outf = gzip.open("{0}{1}.tsv.gzip".format(output_prefix, i), "wt")
    # Sort reviews by their rating and then document ID
    for (doc_idx, score, doc) in sorted(fold_docs, key=itemgetter(1, 0)):
        outf.write("{0}\t{1}\t{2}\n".format(doc_idx, score, doc))
    outf.close()
    del fold_docs
    print("Done fold {0}".format(i))

print("Done")
