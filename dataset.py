import os
import numpy as np
from sklearn.datasets import load_digits


class MNISTDataset():
    def __init__(self, n_classes):
        self.images, self.labels = load_digits(n_class=n_classes, return_X_y=True)
        # Change the scale of data from 0-16 to 0-1
        self.images  = self.images / 16
        # Converting the labels into one hot encoding
        if n_classes == 2:
            self.labels = np.expand_dims(self.labels, axis=1)
        if n_classes > 2:
            labels_one_hot = np.zeros((labels.size, labels.max() + 1))
            labels_one_hot[np.arange(labels.size), labels] = 1
            self.labels = labels_one_hot

    
    def load_data(self):
        return self.images, self.labels

    