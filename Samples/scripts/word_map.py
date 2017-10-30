# Filter some words and assign each word an index.


import os
import string


stop_words = [
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "may",
    "me",
    "might",
    "more",
    "most",
    "must",
    "my",
    "myself",
    "of",
    "off",
    "on",
    "once",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "shall",
    "she",
    "should",
    "some",
    "such",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "under",
    "until",
    "up",
    "us",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves"
]

filter_words = [
]

skip_word_once = True


def to_system_path(path):
    if os.name == "nt":
        return path.replace("/", "\\")
    else:
        return path.replace("\\", "/")


def to_standard_path(path):
    return path.replace("\\", "/")


dir_path = to_standard_path(os.path.dirname(os.path.realpath(__file__)))
dir_path = "/".join(dir_path.split("/")[:-1])
input_path = to_system_path("{0}/words.tsv".format(dir_path))
output_path = to_system_path("{0}/word_map.tsv".format(dir_path))

patterns = []
for c in string.ascii_lowercase[:26]:
    patterns.append("{0}{0}{0}".format(c))

words = []
with open(input_path, "r") as inf:
    for line in inf:
        line = line[:-1]
        if len(line) < 1:
            continue
        components = line.split("\t")
        word = components[0]
        cnt = int(components[1])
        if (cnt == 1) and skip_word_once:
            continue
        if (word in stop_words) or (word in filter_words):
            continue
        if any(pattern in word for pattern in patterns):
            continue
        words.append(word)
inf.close()

outf = open(output_path, "w")
cnt = 1
for word in sorted(words):
    outf.write("{0}\t{1}\n".format(cnt, word))
    cnt += 1
outf.close()

print("Done")
