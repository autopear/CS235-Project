# Randomly split all reviews in to k groups. So we can use k-1 groups for training, and 1 group for testing.
# People usually use 5 groups, which is 5-fold cross validation.


import os
import random
import gzip
import glob
from operator import itemgetter


k = 5


def to_system_path(path):
    if os.name == "nt":
        return path.replace("/", "\\")
    else:
        return path.replace("\\", "/")


def to_standard_path(path):
    return path.replace("\\", "/")


dir_path = to_standard_path(os.path.dirname(os.path.realpath(__file__)))
dir_path = "/".join(dir_path.split("/")[:-1])
input_paths = []
output_prefix = to_system_path("{0}/5-fold/fold-".format(dir_path))

if os.path.isfile(to_system_path("{0}/samples_indices.tsv.gzip".format(dir_path))):
    input_paths.append(to_system_path("{0}/samples_indices.tsv.gzip".format(dir_path)))
else:
    for f in glob.glob(to_system_path("{0}/samples_indices-*.tsv.gzip".format(dir_path))):
        input_paths.append(to_system_path(f))

docs = []
for input_path in input_paths:
    with gzip.open(input_path, "rt") as inf:
        for line in inf:
            line = line[:-1]
            if len(line) < 1:
                continue
            components = line.split("\t")
            doc_idx = int(components[0])
            score = int(components[1])
            doc = components[2]
            docs.append((doc_idx, score, doc))
    inf.close()
    print("Loaded {0}".format(input_path))
print("{0} documents loaded".format(len(docs)))

# Shuffle 3 times
random.shuffle(docs)
random.shuffle(docs)
random.shuffle(docs)

fold_size = int(len(docs) / k)

for i in range(1, k+1):
    fold_docs = docs[0:fold_size]
    del docs[0:fold_size]

    outf = gzip.open("{0}{1}.tsv.gzip".format(output_prefix, i), "wt")
    for (doc_idx, score, doc) in sorted(fold_docs, key=itemgetter(1, 0)):
        outf.write("{0}\t{1}\t{2}\n".format(doc_idx, score, doc))
    outf.close()
    del fold_docs
    print("Done fold {0}".format(i))

print("Done")
