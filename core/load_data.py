import pandas as pd
import numpy as np

## Read the dataset
train = pd.read_csv('./dataset/fashion-mnist_train.csv')
test = pd.read_csv('./dataset/fashion-mnist_test.csv')

## train_data - 60000, 784
## train_label - 60000,
## test_data - 10000, 784
## test_label - 10000,
train_label = train['label']
train_ds = train.drop('label', axis=1)
test_label = test['label']
test_ds = test.drop('label', axis=1)

## changing it to 
## 60000, 28, 28
## 10000, 28, 28
train_data = train_ds.values.reshape(-1, 28, 28)
test_data = test_ds.values.reshape(-1, 28, 28)

train_data = train_data / 255.0
test_data = test_data / 255.0

train_data = train_data.reshape(60000, 28, 28, -1)
test_data = test_data.reshape(10000, 28, 28, -1)
