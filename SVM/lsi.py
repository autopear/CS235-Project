import os
import itertools
import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


if __name__ == "__main__":
    tf_folder_path = os.path.join("tf-idf_data", "tf")
    idf_folder_path = os.path.join("tf-idf_data", "idf")
    label_folder_path = os.path.join("tf-idf_data", "labels")

    lsi_folder_path = os.path.join("tf-idf_data", "lsi")
    if not os.path.exists(lsi_folder_path):
        os.makedirs(lsi_folder_path)

    num_folds = 5
    folds = [1, 2, 3, 4, 5]
    five_fold = set(folds)

    for fold in itertools.combinations(five_fold, len(folds)-1):

        # Bring out respective labels
        is_first_label_load = True
        for num in fold:
            label_fold_path = os.path.join(label_folder_path, "label_fold_" + str(num) + ".npy")

            if is_first_label_load:
                labels = np.load(label_fold_path)
                is_first_label_load = False
            else:
                temp_labels = np.load(label_fold_path)
                labels = np.vstack((labels, temp_labels))

        # Bring out respective tf
        is_first_tf_load = True
        for num in fold:
            tf_fold_path = os.path.join(tf_folder_path, "tf_fold_" + str(num) + ".npz")

            if is_first_tf_load:
                tf_csr = csr_matrix(scipy.sparse.load_npz(tf_fold_path))
                is_first_tf_load = False
            else:
                temp_tf_csr = csr_matrix(scipy.sparse.load_npz(tf_fold_path))
                tf_csr = scipy.sparse.vstack([tf_csr, temp_tf_csr])

        # Bring out respect idf
        fold_str = "_".join(map(str, fold))
        idf_fold_path = os.path.join(idf_folder_path, "idf_fold_" + fold_str + ".npz")
        idf_csr = csr_matrix(scipy.sparse.load_npz(idf_fold_path))

        tf_idf = scipy.sparse.csr_matrix.multiply(tf_csr, idf_csr)

        # Bring out validation tf and labels & compute tf idf for it using the trained idf
        valid_tf_fold_str = ''.join(map(str, five_fold.difference(fold)))
        valid_tf_path = os.path.join(tf_folder_path, "tf_fold_" + valid_tf_fold_str + ".npz")
        valid_tf_csr = scipy.sparse.load_npz(valid_tf_path)
        valid_tf_labels_path = os.path.join(label_folder_path, "label_fold_" + valid_tf_fold_str + ".npy")
        valid_labels = np.load(valid_tf_labels_path)
        valid_tf_idf = scipy.sparse.csr_matrix.multiply(valid_tf_csr, idf_csr)

        # Compute LSI for both
        train_LSI, _, _ = scipy.sparse.linalg.svds(tf_idf, k=100)
        valid_LSI, _, _ = scipy.sparse.linalg.svds(valid_tf_idf, k=100)

        # Attach labels to be the FIRST column
        train_LSI = csr_matrix(train_LSI).todense()
        valid_LSI = csr_matrix(valid_LSI).todense()

        train_LSI = np.hstack([labels, train_LSI])
        valid_LSI = np.hstack([valid_labels, valid_LSI])

        lsi_fold_path = os.path.join(lsi_folder_path, "lsi_fold_" + fold_str)
        valid_lsi_path = os.path.join(lsi_folder_path, "lsi_valid_" + fold_str)

        np.save(lsi_fold_path, train_LSI)
        np.save(valid_lsi_path, valid_LSI)

        print("Saved Train LSI: " + str(lsi_fold_path))
        print("Saved Valid LSI: " + str(valid_lsi_path))
        print("--" * 50)


