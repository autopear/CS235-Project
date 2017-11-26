# scripts

This directory contains all Python scripts to sample reviews, building word indices, and create files for corss validation.

## Required Libraries:
* [matplotlib](https://matplotlib.org/)
* [numpy](http://www.numpy.org/)

### [k-fold.py](k-fold.py)
Create files for cross validation.

### [qqplot.py](qqplot.py)
Q-Q plot for review length distribution per rating.

### [sample_reviews.py](sample_reviews.py)
Random sample reviews.

### [score_dist.py](score_dist.py)
Plot histogram matrices for rating distribution per fold and for all sampled reviews.

### [word_count.py](word_count.py)
Generate a list of words with their frequencies.

### [word_map.py](word_map.py)
Generate indices for selected words.

### [word_to_idx.py](word_to_idx.py)
Convert text reviews to word indices.
