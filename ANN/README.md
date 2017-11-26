ANN
=============

This folder contains codes and results for ANN (Approximate Nearest Neighbor) related word. Since our data size is too big to fit into memory (over 20 GB estimated), we tried KD-Tree but it failed. Our solution is to use Facebook's [PySparNN](https://github.com/facebookresearch/pysparnn) to have approximate predictions.

## Required Libraries:
* [numpy](http://www.numpy.org/)
* [scipy](https://www.scipy.org/)
* [PySparNN](https://github.com/facebookresearch/pysparnn) (Only the class is needed)

## Scripts
* [pysparnn](scripts/pysparnn/): Class library from Facebook's ANN.
* [ann.py](scripts/ann.py): Use ANN algrithm to classify reviews for each test fold. It's split into 4 sub-tasks, binary classification (score 4 and 5 are positive, 1, 2 and 3 are negative) v.s. multiclass classification, and whether the final scores are weighted or not.
* [stats_binary.py](scripts/stats_binary.py): Compute statistics for binary classification for both unweighted and weighted versions.
* [stats_multiclass.py](scripts/stats_multiclass.py): Compute statistics for multiclass classification for both unweighted and weighted versions.

Actual labels (column 1) and predicted labels (column 2) are stored under [output](output/) with the fold number (used as test sets) and script embedded in the filename. Results are also available at [result_multiclass.txt](output/result_multiclass.txt) and [result_binary.txt](output/result_binary.txt).

## License
[BSD](LICENSE)
