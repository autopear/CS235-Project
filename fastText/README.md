fastText
=============

This folder contains codes and results for [fastText](https://github.com/facebookresearch/fastText) algorithm.

fastText accepts text as input, so first you need to revert the 5-fold files back to words. This can be done by [revert_word_index.py](scripts/revert_word_index.py). It will create 5 tsv files (gzipped) under data folder.

fastText can do binary classification and multiclass classification. But to compute precison, recall or F-1 score, labels must be binary. So I provide two scripts.

- [fasttext_all.py](scripts/fasttext_all.py): To compute precison and recall, one label is selected as positive, and all other labels are selected as negative. Then compute precisions and recalls for each label. The average precisions and recalls will be used.

- [fasttext_bin.py](scripts/fasttext_bin.py): This is the binary version. This sets score 4 and 5 as positive, and score 1 to 3 as negative.

Actual labels and predicted labels are stored under [output](output) with the fold number (used as test sets) and script embedded in the filename. Results are also available at [result_all.txt](output/result_all.txt) and [result_bin.txt](output/result_bin.txt).
