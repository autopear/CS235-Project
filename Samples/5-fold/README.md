# 5-fold

This is supposed to be used for 5-fold cross validation. We run the evaluation process 5 times, each time we select 4 groups for training, and 1 group for testing (hold-out set). In the end, we take the average of accuracies as the final result.

For example, in round 1, we can select 1, 2, 3, 4 for training a model, 5 for verification. The next round, we may select 1, 2, 3, 5 for training, and 4 for verification.

Each group has exact 40,000 reviews ordered by score, then document index. No two groups have common reviews (no duplicates).