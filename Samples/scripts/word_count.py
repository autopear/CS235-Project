# Count all word frequencies through all reviews.


import os
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
input_path = to_system_path("{0}/samples.tsv".format(dir_path))
output_path = to_system_path("{0}/words.tsv".format(dir_path))

word_bag = {}

with open(input_path, "r") as inf:
    for line in inf:
        line = line[:-1]
        if len(line) < 1:
            continue
        components = line.split("\t")
        words = components[2].split(" ")
        for word in words:
            if len(word) > 1:
                cnt = word_bag.get(word, 0)
                word_bag[word] = cnt + 1
inf.close()

outf = open(output_path, "w")
# Sort by word count in descending order, then by word in ascending order
for (word, cnt) in sorted(word_bag.items(), key=lambda x: (-x[1], x[0])):
    outf.write("{0}\t{1}\n".format(word, cnt))
outf.close()

print("Done")
