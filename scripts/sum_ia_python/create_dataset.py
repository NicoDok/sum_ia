import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from config import MNIST_TEST_CSV, MNIST_TRAIN_CSV, DATASET


def load_data(path_csv):
    df_matrix = pd.read_csv(path_csv, header=None)  # data_frame
    labels = df_matrix.iloc[:, 0]  # iloc "indice" ; loc "identifiant striyesng" ; iloc + rapide que loc
    labels = labels.values.astype(np.uint8)  # transforme en array numpy
    images = df_matrix.iloc[:, 1:].values.astype(np.uint8).reshape(-1, 28, 28)  # desapplati ...
    n = labels.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    idx1 = idx[::2]
    idx2 = idx[1::2]
    matrix_y_ = labels[idx1] + labels[idx2]
    enc = OneHotEncoder(handle_unknown='ignore')
    matrix_y = enc.fit_transform(matrix_y_.reshape(-1, 1)).todense()  # vecteur de 0 sauf 1 sur le resultat
    matrix_x = np.concatenate([images[idx1, ...], images[idx2, ...]], axis=2)  # concatene images sur le 2e axe
    matrix_x = matrix_x.reshape(-1, 28 * 56)  # re applatir pour le réseau
    return matrix_x, matrix_y, matrix_y_


def main():
    x, y, label = load_data(MNIST_TRAIN_CSV)
    test_x, test_y, test_label = load_data(MNIST_TEST_CSV)
    n = x.shape[0]
    n_train = n - (n // 3)
    np.savez(DATASET,
             train_x=x[:n_train, :],
             train_y=y[:n_train, :],
             train_label=label[:n_train],
             validation_x=x[n_train:, :],
             validation_y=y[n_train:, :],
             validation_label=label[n_train:],
             test_x=test_x,
             test_y=test_y,
             test_label=test_label
             )


if __name__ == "__main__":
    main()
