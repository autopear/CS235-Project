# Based on the word index map, convert all reviews to use word indices. Words without index will be skipped.


import glob
import gzip
import os


min_num_words = 100  # Minimum number of words per review to be kept


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
backup_path = to_system_path("{0}/samples_backup.tsv.gzip".format(dir_path))
map_path = to_system_path("{0}/word_map.tsv".format(dir_path))

word_map = {}
with open(map_path, "r") as inf:
    for line in inf:
        line = line[:-1]
        if len(line) < 1:
            continue
        components = line.split("\t")
        idx = int(components[0])
        word = components[1]
        word_map[word] = idx
inf.close()
print("Word mapping loaded")

backups = [[]] * 5
with gzip.open(backup_path, "rt") as inf:
    for line in inf:
        line = line[:-1]
        if len(line) < 1:
            continue
        components = line.split("\t")
        doc_idx = int(components[0])
        score = int(components[1])
        words = components[2].split(" ")
        indices = []
        for word in words:
            idx = word_map.get(word, -1)
            if idx > 0:
                indices.append(str(idx))
        if len(indices) < min_num_words:
            continue
        t = backups[score-1]
        t.append("{0}\t{1}\t{2}\n".format(doc_idx, score, " ".join(indices)))
        backups[score-1] = t
inf.close()
print("Backup loaded")


def save_file(inp, outp):
    """ Convert reviews from text to word indices, fill up with backup reviews if needed """
    outf = gzip.open(outp, "wt")
    with gzip.open(inp, "rt") as inf:
        for line in inf:
            line = line[:-1]
            if len(line) < 1:
                continue
            components = line.split("\t")
            doc_idx = int(components[0])
            score = int(components[1])
            words = components[2].split(" ")
            indices = []
            for word in words:
                idx = word_map.get(word, -1)
                if idx > 0:
                    indices.append(str(idx))
            if len(indices) > min_num_words:
                outf.write("{0}\t{1}\t{2}\n".format(doc_idx, score, " ".join(indices)))
            else:
                # If the review does not satisfy the requirement, find one review in the backup as a replacement
                t = backups[score-1]
                if len(t) > 0:
                    outf.write(t[0])
                    del t[0]
                    backups[score-1] = t
                else:
                    print("No backup available")
                    break
    outf.close()
    print("Saved {0}".format(outp))


input_path = to_system_path("{0}/samples.tsv.gzip".format(dir_path))
output_path = to_system_path("{0}/samples_indices.tsv.gzip".format(dir_path))
if os.path.isfile(input_path):
    save_file(input_path, output_path)
else:
    for input_path in glob.glob(to_system_path("{0}/samples-*.tsv.gzip".format(dir_path))):
        fn = to_standard_path(input_path).split("/")[-1]
        fn = fn.replace("samples-", "samples_indices-")
        output_path = to_system_path("{0}/{1}".format(dir_path, fn))
        save_file(input_path, output_path)

print("Done")
