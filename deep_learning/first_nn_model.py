import numpy as np
import pandas as pd

from sklearn.preprocessing import label_binarize


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
 expA = np.exp(x)
 return expA / expA.sum(axis = 1)


def relu(x):
    return x * (x > 0)


def get_z(w, x):
    return relu(x.dot(w))


def get_y(v, z, s_prime):
    return s_prime(z.dot(v))


def forward(x, w, v, s_prime=sigmoid):
    z = get_z(w, x)
    y = get_y(v, z, s_prime)
    return y


def load_csv(csv_dir, target_label, feat_labels=None):
    df = pd.read_csv(csv_dir)
    target = df[target_label]
    if feat_labels is None:
        feature = df.drop(target_label, 1)
    else:
        feature = df[feat_labels]
    return feature, target


def y2indicator(y):
    return label_binarize(y, classes=np.unique(y)).transpose()[0]


def cost(t, y):
    return -(t*np.log(y)).sum()
