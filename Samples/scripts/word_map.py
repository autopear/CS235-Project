# Filter some words and assign each word an index.


import os
import string


# Manually add some words to drop
filter_words = [
]

min_word_freq = 3  # A word must appear at least 3 times to be kept


def to_system_path(path):
    """ Convert an input path to the current system style, \ for Windows, / for others """
    if os.name == "nt":
        return path.replace("/", "\\")
    else:
        return path.replace("\\", "/")


def to_standard_path(path):
    """ Convert \ to \ in path (mainly for Windows) """
    return path.replace("\\", "/")


dir_path = to_standard_path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # Module folder
input_path = to_system_path("{0}/words.tsv".format(dir_path))
output_path = to_system_path("{0}/word_map.tsv".format(dir_path))

words = []
with open(input_path, "r") as inf:
    for line in inf:
        line = line[:-1]
        if len(line) < 1:
            continue
        components = line.split("\t")
        word = components[0]
        cnt = int(components[1])
        if cnt < min_word_freq:
            continue
        if word in filter_words:
            continue
        words.append(word)
inf.close()
print("{0} words loaded".format(len(words)))

outf = open(output_path, "w")
cnt = 1
for word in sorted(words):
    outf.write("{0}\t{1}\n".format(cnt, word))
    cnt += 1
outf.close()

print("Done")
