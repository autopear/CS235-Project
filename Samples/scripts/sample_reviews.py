# Randomly sample reviews following the same score distribution.
# A backup file is also saved incase some sampled reviews get removed due to removal of infrequent words


import os
import random
import string
import gzip
import math
from operator import itemgetter

sample_size = 200000
num_batches = 2  # Number of split files to store the reviews. Github allows 100 MB per file only.
min_num_words = 100  # Selected reviews must have at least 100 words
backup_ratio = 0.05  # Keep extra 5% of reviews for backup in case some reviews are dropped in the processing


def to_system_path(path):
    """ Convert an input path to the current system style, \ for Windows, / for others """
    if os.name == "nt":
        return path.replace("/", "\\")
    else:
        return path.replace("\\", "/")


def to_standard_path(path):
    """ Convert \ to / in path (mainly for Windows) """
    return path.replace("\\", "/")


stop_words = (
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
)

# Construct a list of patterns of aaa, bbb, ccc, ..., zzz
three_chars = []
for c in string.ascii_lowercase[:26]:
    three_chars.append("{0}{0}{0}".format(c))


def contains_three_chars(s):
    """ Check if a string contains consecutive 3 the same characters """
    if any(cs in s for cs in three_chars):
        return True
    else:
        return False


def process_document(sentence):
    """ Remove stop words, remove too short words, remove words that contain 3 consecutive characters
        Remove too short documents after processing """
    r = []
    for word in sentence.split(" "):
        if (len(word)) < 2 or (word in stop_words) or (contains_three_chars(word)):
            continue
        else:
            r.append(word)
    if len(r) < min_num_words:
        return ""
    else:
        return " ".join(r)


dir_path = to_standard_path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # Module directory
input_path = to_system_path("{0}/Preprocessing/output/stemmed.tsv".format(os.path.dirname(dir_path)))
sample_prefix = to_system_path("{0}/samples".format(dir_path))
backup_path = to_system_path("{0}/samples_backup.tsv.gzip".format(dir_path))  # Backup reviews
log_path = to_system_path("{0}/stats.txt".format(dir_path))  # Statistics of sampling

# Read and process all reviews, save them according to their ratings
reviews = {
    1: [],
    2: [],
    3: [],
    4: [],
    5: []
}

with open(input_path, "r") as inf:
    ln = 1
    for line in inf:
        line = line[:-1]
        if len(line) < 1:
            continue
        components = line.split("\t")
        doc_idx = int(components[0])
        score = int(components[1])
        review = process_document(components[2])
        if len(review) > 0:
            t = reviews[score]
            t.append((doc_idx, review))
            reviews[score] = t
        ln += 1
inf.close()

counts = {}  # Number of reviews per rating
num_reviews = 0
for score in range(1, 6):
    cnt = len(reviews[score])
    counts[score] = cnt
    num_reviews += cnt
print("Total: {0}\n{1}".format(num_reviews, counts))

if num_reviews <= round(sample_size * (1 + backup_ratio)):
    print("Not enough reviews for sampling")
else:
    logf = open(log_path, "w")
    logf.write("Total # reviews: {0}\n".format(num_reviews))

    bakf = gzip.open(backup_path, "wt")
    total_samples = 0

    batch_size = math.floor(float(sample_size) / num_batches)  # Number of reviews per batch file

    batch_id = -1
    outf = None
    if num_batches > 1:
        batch_id = 1
        outf = gzip.open("{0}-{1}.tsv.gzip".format(sample_prefix, batch_id), "wt")
    else:
        outf = gzip.open("{0}.tsv.gzip".format(sample_prefix), "wt")

    lines_written = 0

    ks = 0

    for score in range(1, 6):
        cnt = counts[score]
        k = round(float(cnt) * sample_size / num_reviews)  # Number of reviews per rating
        if score == 5:
            k = sample_size - ks  # Fix rounding error
        else:
            ks += k
        t = round(backup_ratio * k)  # Number of backup reviews per rating
        total_samples += k
        logf.write("Score {0}: {1} / {2}\n".format(score, k, cnt))
        samples = random.sample(reviews[score], t+k)  # Sample enough reviews per rating
        # Write first k reviews for normal file
        for doc_idx, review in sorted(samples[0:k], key=itemgetter(0)):
            outf.write("{0}\t{1}\t{2}\n".format(doc_idx, score, review))
            lines_written += 1
            if (num_batches > 1) and (lines_written % batch_size == 0):
                outf.close()
                outf = None
                if lines_written < sample_size:
                    batch_id += 1
                    outf = gzip.open("{0}-{1}.tsv.gzip".format(sample_prefix, batch_id), "wt")
        # Write the last t reviews for backup
        for doc_idx, review in sorted(samples[k:t+k], key=itemgetter(0)):
            bakf.write("{0}\t{1}\t{2}\n".format(doc_idx, score, review))
        print("Done for score {0}".format(score))

    logf.write("Total samples: {0}\n".format(total_samples))

    logf.close()
    if outf:
        outf.close()
    bakf.close()

    print("Done")
