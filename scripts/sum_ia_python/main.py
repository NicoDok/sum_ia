#!/usr/bin/env python
# coding: utf-8

from torch import nn, FloatTensor, LongTensor, optim, autograd

from utils import read_dataset
from network import Network


def main():
    data = read_dataset()

    net = Network()

    # quelle fonction de descente de gradient ? rapidité de descente sur l'erreur
    optimizer = optim.SGD(net.parameters(), lr=0.00005, momentum=0.9)
    loss_func = nn.CrossEntropyLoss()

    x = FloatTensor(data.train_x)
    y = LongTensor(data.train_label)

    loss_log = []

    batch_size = 10
    n = data.train_x.shape[0]

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


if __name__ == "__main__":
    main()
