
import os
import random

sample_size = 200000


def to_system_path(path):
    if os.name == "nt":
        return path.replace("/", "\\")
    else:
        return path.replace("\\", "/")


def to_standard_path(path):
    return path.replace("\\", "/")


dir_path = to_standard_path(os.path.dirname(os.path.realpath(__file__)))
input_path = to_system_path("{0}/Preprocessing/output/stemmed.tsv".format("/".join(dir_path.split("/")[:-2])))
sample_path = to_system_path("{0}/samples.tsv".format("/".join(dir_path.split("/")[:-1])))
log_path = to_system_path("{0}/stats.txt".format("/".join(dir_path.split("/")[:-1])))

reviews = {
    1: [],
    2: [],
    3: [],
    4: [],
    5: []
}

with open(input_path, "r") as inf:
    for line in inf:
        if len(line) < 1:
            continue
        components = line.split("\t")
        score = int(components[1])
        t = reviews[score]
        t.append(line)
        reviews[score] = t
inf.close()

counts = {}
num_reviews = 0
for score in range(1, 6):
    cnt = len(reviews[score])
    counts[score] = cnt
    num_reviews += cnt
print("Total: {0}\n{1}".format(num_reviews, counts))

if num_reviews <= sample_size:
    print("Not enough reviews for sampling")
else:
    logf = open(log_path, "w")
    logf.write("Total # reviews: {0}\n".format(num_reviews))

    outf = open(sample_path, "w")
    total_samples = 0

    for score in range(1, 6):
        cnt = counts[score]
        k = round(float(cnt) * sample_size / num_reviews)
        total_samples += k
        logf.write("Score {0}: {1} / {2}\n".format(score, k, cnt))
        samples = random.sample(reviews[score], k)
        outf.write("".join(samples))
        print("Done for score {0}".format(score))

    logf.write("Total samples: {0}\n".format(total_samples))

    logf.close()
    outf.close()

    if os.name != "nt":
        print("Sorting...")
        os.system("sort -n -k2,2 -k1,1 \"{0}\" > \"{0}.sorttmp\" && mv -f \"{0}.sorttmp\" \"{0}\"".format(sample_path))

    print("Done")
