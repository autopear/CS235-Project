
import os


def to_system_path(path):
    if os.name == "nt":
        return path.replace("/", "\\")
    else:
        return path.replace("\\", "/")


def to_standard_path(path):
    return path.replace("\\", "/")


dir_path = to_standard_path(os.path.dirname(os.path.realpath(__file__)))
input_path = to_system_path("{0}/samples.tsv".format("/".join(dir_path.split("/")[:-1])))
map_path = to_system_path("{0}/word_map.tsv".format("/".join(dir_path.split("/")[:-1])))
output_path = to_system_path("{0}/samples_indices.tsv".format("/".join(dir_path.split("/")[:-1])))

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

outf = open(output_path, "w")
with open(input_path, "r") as inf:
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
        outf.write("{0}\t{1}\t{2}\n".format(doc_idx, score, " ".join(indices)))
outf.close()

print("Done")
