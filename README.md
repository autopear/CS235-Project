Sentiment Analysis Based on Amazon Reviews
=============

## [Preprocessing](Preprocessing)
We downloaded a review dataset of "Movies and TV" from [http://jmcauley.ucsd.edu/data/amazon](http://jmcauley.ucsd.edu/data/amazon). Here we preprocessed and stemmed the dataset as the input to other programs.

## [Samples](Samples)
The original dataset contains 1,697,533 reviews. We randomly sampled 200,000 reviews with constraints and created 5 folds for cross validation.

## [Naive-Bayes](Naive-Bayes)
Naïve Bayes is one of the most classical classification algorithm that are widely used in natural language processing. It natively supports multi-class classification tasks.

## [SVM](SVM)
SVM is commonly used in natural language processing tasks. We applied SVM on the *tf-idf* model of our dataset for classification.

## [ANN](ANN)
ANN stands for Approximate Nearest Neighbor. It reduces time and storage requirement to compute nearest neighbors with high dimensional data.

## [doc2vec](doc2vec)
doc2vec can convert text documents into fixed length vectors in a small dimensional space. Using these vectors we are able to perform different SVM and *k*NN classification other than *tf-idf*.

## [fastText](fastText)
Facebook's fastText utilizes deep neural network based on an improved version of word2vec model. It provides functions to classifiy text documents in a very efficient manner.

## License
BSD, LGPL
