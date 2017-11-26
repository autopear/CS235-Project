# This script extract review text, summary and score from the json file
# The output file format is a tsv file with 4 columns: document index, score, summary and review.

import gzip
import html
import json
import os
import shutil
import urllib.request


input_file = "reviews_Movies_and_TV_5.json.gz"
url = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz"


def trim_spaces(txt):
    """ Remove consecutive spaces and new line characters """
    tmp = txt.replace("\t", " ").replace("\r", " ").replace("\n", " ").replace("\\r", " ").replace("\\n", " ").strip()
    while "  " in tmp:
        tmp = tmp.replace("  ", " ")
    return tmp


def to_system_path(path):
    """ Convert an input path to the current system style, \ for Windows, / for others """
    if os.name == "nt":
        return path.replace("/", "\\")
    else:
        return path.replace("\\", "/")


def to_standard_path(path):
    """ Convert \ to \ in path (mainly for Windows) """
    return path.replace("\\", "/")


dir_path = to_standard_path(os.path.dirname(os.path.realpath(__file__)))  # Module folder
input_path = to_system_path("{0}/{1}".format(dir_path, input_file))

# Download if not exists
if not os.path.isfile(input_path):
    with urllib.request.urlopen(url) as src, open(input_path, 'wb') as dest:
        shutil.copyfileobj(src, dest)
    dest.close()
    print("{0} downloaded".format(input_file))

output_path = to_system_path("{0}/output/extracted.tsv".format(dir_path))
outf = open(output_path, "w")

reviews_written = 0
reviews_skipped = 0

print("Reading json file")
with gzip.open(input_path, "rt") as jsf:
    for line in jsf:
        line = line[:-1]  # Trim trailing new line chars
        if len(line) < 1:
            continue  # Skip empty lines
        json_dict = json.loads(line)
        score = int(json_dict.get("overall", 0))
        if score < 1:
            reviews_skipped += 1
            continue  # Skip reviews without a score
        summary = trim_spaces(html.unescape(json_dict.get("summary", "")))
        review = trim_spaces(html.unescape(json_dict.get("reviewText", "")))

        if len(summary) + len(review) < 100:
            reviews_skipped += 1
            continue  # Skip too short reviews

        output_line = "{0}\t{1}\t{2}\t{3}\n".format(reviews_written+1, score, summary, review)
        outf.write(output_line)
        reviews_written += 1
jsf.close()
outf.close()

log_file = open(to_system_path("{0}/output/extracted.log".format(dir_path)), "w")
log_file.write("Written: {0}\nSkipped: {1}".format(reviews_written, reviews_skipped))
log_file.close()

print("Done")
