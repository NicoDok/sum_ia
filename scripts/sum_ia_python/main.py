#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from torch import nn, FloatTensor, LongTensor, optim, autograd

from config import MNIST_TEST_CSV, MNIST_TRAIN_CSV
from network import Network

def load_data(path_csv):
    df_matrix = pd.read_csv(path_csv, header=None) # data_frame
    labels = df_matrix.iloc[:, 0] # iloc "indice" ; loc "identifiant striyesng" ; iloc + rapide que loc
    labels = labels.values.astype(np.uint8) # transforme en array numpy
    images = df_matrix.iloc[:, 1:].values.astype(np.uint8).reshape(-1, 28, 28) # desapplati ...
    n = labels.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    idx1 = idx[::2]
    idx2 = idx[1::2]
    matrix_y_ = labels[idx1] + labels[idx2]
    enc = OneHotEncoder(handle_unknown='ignore')
    matrix_y = enc.fit_transform((labels[idx1] + labels[idx2]).reshape(-1, 1)).todense() # vecteur de 0 sauf 1 sur le resultat
    matrix_x = np.concatenate([images[idx1, ...], images[idx2, ...]], axis=2) # concatene images sur le 2e axe
    matrix_x = matrix_x.reshape(-1, 28*56) # re applatir pour le réseau
    return matrix_x, matrix_y, matrix_y_

train_x, train_y, train_y_ = load_data(MNIST_TRAIN_CSV)
val_x, val_y, val_y_ = load_data(MNIST_TEST_CSV)

net = Network()

optimizer = optim.SGD(net.parameters(), lr=0.00005, momentum=0.9) # quelle fonction de descente de gradient ? rapidité de descente sur l'erreur
loss_func = nn.CrossEntropyLoss()

x = FloatTensor(train_x)
y = LongTensor(train_y_)

loss_log = []

batch_size = 10
n = train_x.shape[0]

for e in range(20):
    for i in range(0, n, batch_size):
        x_mini = x[i:i + batch_size] 
        y_mini = y[i:i + batch_size] 
        
        x_var = autograd.Variable(x_mini)
        y_var = autograd.Variable(y_mini)
        
        optimizer.zero_grad()
        net_out = net(x_var)
        
        loss = loss_func(net_out, y_var)
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            loss_log.append(loss.item())
        
    print(f'Epoch: {e} - Loss: {loss.item():.6f}')
