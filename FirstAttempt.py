#!/usr/bin/env python
# coding: utf-8

# In[215]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

mnist_train_csv = "../data/mnist_train.csv"
mnist_test_csv = "../data/mnist_test.csv"

df_train = pd.read_csv(mnist_train_csv, header=None) # data_frame


# In[216]:


labels = df_train.iloc[:, 0] # iloc "indice" ; loc "identifiant string" ; iloc + rapide que loc
labels = labels.values.astype(np.uint8) # transforme en array numpy

labels


# In[217]:

images = df_train.iloc[:, 1:].values.astype(np.uint8).reshape(-1, 28, 28) # desapplati ...
# images


# In[218]:


n = labels.shape[0]
idx = np.arange(n)
np.random.shuffle(idx)
idx1 = idx[::2]
idx2 = idx[1::2]
idx1


# In[219]:


images[idx1, ...].shape


# In[220]:


train_y_ = labels[idx1] + labels[idx2]
train_y_


# In[221]:


enc = OneHotEncoder(handle_unknown='ignore')
train_y = enc.fit_transform((labels[idx1] + labels[idx2]).reshape(-1, 1)).todense() # vecteur de 0 sauf 1 sur le resultat
train_y.shape


# In[222]:


train_x = np.concatenate([images[idx1, ...], images[idx2, ...]], axis=2) # concatene images sur le 2e axe


# In[223]:


train_x = train_x.reshape(-1, 28*56) # re applatir pour le réseau


# In[224]:


train_x[0, :].reshape(28, 56)


# In[225]:


df_test = pd.read_csv(mnist_test_csv, header=None) # data_frame


# In[226]:


labels = df_train.iloc[:, 0] # iloc "indice" ; loc "identifiant string" ; iloc + rapide que loc
labels = labels.values.astype(np.uint8) # transforme en array numpy
images = df_train.iloc[:, 1:].values.astype(np.uint8).reshape(-1, 28, 28) # desapplati ...
n = labels.shape[0]
idx = np.arange(n)
np.random.shuffle(idx)
idx1 = idx[::2]
idx2 = idx[1::2]
val_y_ = labels[idx1] + labels[idx2]
enc = OneHotEncoder(handle_unknown='ignore')
val_y = enc.fit_transform((labels[idx1] + labels[idx2]).reshape(-1, 1)).todense()
val_y.shape

val_x = np.concatenate([images[idx1, ...], images[idx2, ...]], axis=2) # concatene images sur le 2e axe
val_x.reshape(-1, 28*56).shape # re applatir pour le réseau


# In[227]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


# In[228]:


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.l1 = nn.Linear(1568, 392)
        self.relu1 = nn.ReLU() # casse la linearite
        self.l2 = nn.Linear(392, 98)
        self.relu2 = nn.ReLU() # casse la linearite
        self.l3 = nn.Linear(98, 19)
        
    def forward(self, x):
        x = self.l1(x)
        x = self.relu1(x)
        x = self.l2(x)
        x = self.relu2(x)
        x = self.l3(x)
        return F.softmax(x, dim=1)


# In[229]:


net = Network()


# In[230]:


optimizer = optim.SGD(net.parameters(), lr=0.00005, momentum=0.9) # quelle fonction de descente de gradient ? rapidité de descente sur l'erreur
loss_func = nn.CrossEntropyLoss()


# In[231]:


train_x.shape


# In[235]:


x = torch.FloatTensor(train_x)
y = torch.LongTensor(train_y_)

loss_log = []

batch_size = 10
n = train_x.shape[0]

for e in range(20):
    for i in range(0, n, batch_size):
        x_mini = x[i:i + batch_size] 
        y_mini = y[i:i + batch_size] 
        
        x_var = Variable(x_mini)
        y_var = Variable(y_mini)
        
        optimizer.zero_grad()
        net_out = net(x_var)
        
        loss = loss_func(net_out, y_var)
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            loss_log.append(loss.item())
        
    print(f'Epoch: {e} - Loss: {loss.item():.6f}')
