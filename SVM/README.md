## tf-idf SVM
There are two implementations of the SVM.
One implementation uses the Pegasos algorithm to optimize the support vector.
It also includes a linear and rbf kernel.
The other implementation is from the `sklearn` package.

## tf-idf.py
Samples from each fold to compute the tf and idf of the data.
The matrices are stored in `tf-idf_data/tf` and `tf-idf_data/idf` for each fold respectively.
Each `.npz` file has the content of the numbered folds in its name.
** This script takes a long time to run on the data set.

## lsi.py
Computes the tf-idf of the matrices from the above file, `tf-idf.py`.
Singular value decomposition is then applied to the matrices using `sklearn`.
The respective training and validation folds are stored in `tf-idf_data/lsi`.
Files with `fold` in the name consists of the tf-idf of the respective folds.
Files with `valid` in the name consists of the tf of the left out fold and the
idf of the training folds.

## train_svm.py
Uses the matrices in `tf-idf_data/lsi` to train the SVM.
The training is done using the Pegasos algorithm.
The weights or support vectors are stored in `tf-idf_data/sv`.

## train_skl_svm.py
Uses the matrices in `tf-idf_data/lsi` to train the SVM.
This uses the `sklearn` implementation, SVC.
