# Score distribution for the whole samples and 5 groups


import gzip
import itertools
import matplotlib.pyplot as plt
import numpy
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


dir_path = to_standard_path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # Module directory
dir_path = "{0}/5-fold".format(dir_path)  # 5 fold directory
input_prefix = to_system_path("{0}/fold-".format(dir_path))

scores = []  # Number of reviews per score per fold
overall_scores = [0, 0, 0, 0, 0]  # Number of reviews per rating in the whole sample

for i in range(0, 5):
    input_path = "{0}{1}.tsv.gzip".format(input_prefix, i+1)
    group_scores = [0, 0, 0, 0, 0]
    with gzip.open(input_path, "rt") as inf:
        for line in inf:
            line = line[:-1]
            if len(line) < 1:
                continue
            score = int(line.split("\t")[1])
            group_scores[score-1] = group_scores[score-1] + 1
    inf.close()
    del inf
    t = sum(group_scores)
    for j in range(0, 5):
        overall_scores[j] = overall_scores[j] + group_scores[j]
        group_scores[j] = float(group_scores[j]) / t
    scores.append(group_scores)

print("Scores loaded")

t = sum(overall_scores)
for i in range(0, 5):
    overall_scores[i] = float(overall_scores[i]) / t

fig, axes = plt.subplots(nrows=3, ncols=2)
ax0, ax1, ax2, ax3, ax4, ax5 = axes.flatten()

x_labels = ("1 Star", "2 Stars", "3 Star", "4 Stars", "5 Stars")
y_labels = ("0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%")

y_pos = numpy.arange(5)


def subplot(ax, nums, title, alpha=0.6):
    rects = ax.bar(y_pos, nums, align="center", alpha=alpha)

    ax.set_xticks([0, 1, 2, 3, 4], minor=False)
    ax.set_xticklabels(x_labels, minor=False)

    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], minor=False)
    ax.set_yticklabels(y_labels, minor=False)

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.03 * height,
                "{:.1%}".format(height),
                ha="center", va="bottom",
                fontsize="smaller")

    ax.set_title(title)


subplot(ax0, overall_scores, "Overall", 1)
subplot(ax1, scores[0], "Fold 1")
subplot(ax2, scores[1], "Fold 2")
subplot(ax3, scores[2], "Fold 3")
subplot(ax4, scores[3], "Fold 4")
subplot(ax5, scores[4], "Fold 5")

fig.tight_layout()
plt.show()

print("Done")
