fastText
=============

This folder contains codes and results for [fastText](https://github.com/facebookresearch/fastText) algorithm.

fastText accepts text as input, so first you need to revert the 5-fold files back to words. This can be done by [revert_word_index.py](scripts/revert_word_index.py). It will create 5 tsv files (gzipped) under [data](data/) folder.

fastText can do binary classification and multiclass classification. But to compute precison, recall or F-1 score, labels must be binary. So I provide two scripts.

- [fasttext_multiclass.py](scripts/fasttext_multiclass.py): To compute precison and recall, one label is selected as positive, and all other labels are selected as negative. Then compute precisions and recalls for each label. The average precisions and recalls will be used.

- [fasttext_binary.py](scripts/fasttext_binary.py): This is the binary version. This sets score 4 and 5 as positive (+), and score 1 to 3 as negative (-).

Actual labels (column 1) and predicted labels (column 2) are stored under [output](output/) with the fold number (used as test sets) and script embedded in the filename. Results are also available at [result_multiclass.txt](output/result_multiclass.txt) and [result_binary.txt](output/result_binary.txt).
