# Samples

We randomly sampled 200,000 reviews from all reviews, maintaining the original ratio in each group (1 star to 5 stars). So reviews in the sampled set have the same distribution as the original set.

## [samples.tsv](samples.zip)
The sampled 200,000 reviews. Column 1 to 3 are: Document index, score and review. The file is sorted by score and then document index in ascending order.

## [stats.txt](stats.txt)
Shows statistics of the original reviews.

## [words.tsv](words.tsv)
All words from the sampled reviews, with their counts (frequencies) as column 2.

## [word_map.tsv](word_map.tsv)
Selected words and their indices. Some stop words were ignored. And a word containing consecutively 3 same characters will also be ignored (e.g., a word containing aaa, bbb, ..., zzz). These ignored words do not have indices.
Each word is assigned an index as column 1 according to its alphabetic order.
For more details, please see [word_map.py](scripts/word_map.py).

## [samples_indices.tsv](samples_indices.zip)
Based on the word map, all reviews are converted to use word indices. Words that do not have indices will be skipped.

## [5-fold](5-fold/)
The 200,000 reviews are splited into 5 equal groups randomly, each has 40,000 reviews. For more details, please go into the folder.