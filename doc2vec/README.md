doc2vec
=============

This folder contains codes and results for [doc2vec](https://radimrehurek.com/gensim/models/doc2vec.html) algorithm.

doc2vec is one class from [gensim](https://radimrehurek.com/gensim/), which is based on [word2vec](https://www.tensorflow.org/tutorials/word2vec) to convert text documents into fixed length of vectors.

- [doc2vec.py](scripts/doc2vec.py): This script converts all reviews in text format to vectors of size 100.

- [knn.py](scripts/knn.py): kNN binary and multiclass classifier based on doc2vec. [KD-Tree](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html) is used for searcing neighbors.

- [svm.py](scripts/svm.py): SVM binary and multiclass classifier based on doc2vec.

Results are saved as [result_binary.txt](output-knn/result_binary.txt) and [result_multiclass.txt](output-knn/result_multiclass.txt) for kNN classifier, and [result_binary.txt](output-svm/result_binary.txt) and [result_multiclass.txt](output-svm/result_multiclass.txt) for SVM classifier.

## License
[GNU LGPLv2.1 license](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html), BSD
