# Q-Q plot of document length


import glob
import gzip
import matplotlib.pyplot as plt
import os


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

lengths1 = []  # All lengths for fold 1
lengths2 = []  # All lengths for fold 2
lengths3 = []  # All lengths for fold 3
lengths4 = []  # All lengths for fold 4
lengths5 = []  # All lengths for fold 5

for input_path in glob.glob(to_system_path("{0}/samples_indices-*.tsv.gzip").format(dir_path)):
    with gzip.open(input_path, "rt") as inf:
        for line in inf:
            line = line[:-1]
            if len(line) < 1:
                continue
            components = line.split("\t")
            if len(components) != 3:
                continue
            score = int(components[1])
            words = components[2].split(" ")
            wl = len(words)  # Number of words in review
            if score == 1:
                lengths1.append(wl)
            elif score == 2:
                lengths2.append(wl)
            elif score == 3:
                lengths3.append(wl)
            elif score == 4:
                lengths4.append(wl)
            else:
                lengths5.append(wl)
    inf.close()

lengths1.sort()
lengths2.sort()
lengths3.sort()
lengths4.sort()
lengths5.sort()


def get_qq(a_list):
    # Get percentile values (every 5%)
    l = len(a_list)
    ret = []
    for i in range(0, 21):
        idx = int(float(l-1) * i / 20)
        ret.append(a_list[idx])
    return ret


qs1 = get_qq(lengths1)  # Quantiles of fold 1
qs2 = get_qq(lengths2)  # Quantiles of fold 2
qs3 = get_qq(lengths3)  # Quantiles of fold 3
qs4 = get_qq(lengths4)  # Quantiles of fold 4
qs5 = get_qq(lengths5)  # Quantiles of fold 5

# Make 21 (0, 5, 10, ..., 95, 100) x labels
xs = list(range(0, 21))
xlabels = ["{0}%".format(x * 5) for x in xs]

fig, ax = plt.subplots()

line1, = ax.plot(xs, qs1, color="#0000A0", linewidth=1, marker="v", label="1 Star")
line2, = ax.plot(xs, qs2, color="#00FFFF", linewidth=1, marker="s", label="2 Stars")
line3, = ax.plot(xs, qs3, color="#808080", linewidth=1, marker="x", label="3 Stars")
line4, = ax.plot(xs, qs4, color="#FFA500", linewidth=1, marker="d", label="4 Stars")
line5, = ax.plot(xs, qs5, color="#800000", linewidth=1, marker="^", label="5 Stars")

ax.set_xticks(xs)
ax.set_xticklabels(xlabels)
ax.set_ylabel("Document Length")

handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, labels)

plt.show()

print("Done")
