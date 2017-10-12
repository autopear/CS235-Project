# This script extract review text, summary and score from the json file
# The output file format is a tsv file with 4 columns: document index, score, summary and review.

import json
import os
import html


input_file = "reviews_Movies_and_TV_5.json"


def trim_spaces(txt):
    # Remove consecutive spaces and new line characters
    tmp = txt.replace("\t", " ").replace("\r", " ").replace("\n", " ").replace("\\r", " ").replace("\\n", " ").strip()
    while "  " in tmp:
        tmp = tmp.replace("  ", " ")
    return tmp


def to_system_path(path):
    if os.name == "nt":
        return path.replace("/", "\\")
    else:
        return path.replace("\\", "/")


def to_standard_path(path):
    return path.replace("\\", "/")


dir_path = os.path.dirname(os.path.realpath(__file__))
input_path = to_system_path("{0}/{1}".format("/".join(to_standard_path(dir_path).split("/")[0:-1]), input_file))
if os.path.isfile(input_path):
    output_path = to_system_path("{0}/output/extracted.tsv".format(dir_path))
    out_file = open(output_path, "w")

    reviews_written = 0
    reviews_skipped = 0

    with open(input_path, "r") as json_file:
        for line in json_file:
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
            out_file.write(output_line)
            reviews_written += 1

    json_file.close()
    out_file.close()

    log_file = open(to_system_path("{0}/output/extracted.log".format(dir_path)), "w")
    log_file.write("Written: {0}\nSkipped: {1}".format(reviews_written, reviews_skipped))
    log_file.close()

    print("Done")

    print("You may zip the output file into multiple archives (Github does not support large file over 100MB")
else:
    print("Input file not found")
