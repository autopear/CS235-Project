# scripts

This directory contains all Python scripts for ANN (Approximate Nearest Neighbor) related work. Since our data is too big to fit into memory (over 20 GB estimated), we use Facebook's [PySparNN](https://github.com/facebookresearch/pysparnn) method as a replacement for kNN.

## Required Libraries:
* [numpy](http://www.numpy.org/)
* [scipy](https://www.scipy.org/)

### [pysparnn](pysparnn/)
Class library from Facebook's ANN.

### [ann.py](ann.py)
Use ANN algrithm to classify reviews for each test fold. It's split into 4 sub-tasks, binary classification (score 4 and 5 are positive, 1, 2 and 3 are negative) v.s. multiclass classification, and whether the final scores are weighted or not.

### [stats_binary.py](stats_binary.py)
Compute statistics for binary classification for both unweighted and weighted versions.

### [stats_multiclass.py](stats_multiclass.py)
Compute statistics for multiclass classification for both unweighted and weighted versions.

## License
[BSD](../LICENSE)
