import numpy as np
from typing import List
from dense_layer import Dense, Intializer
from losses import Losses, BinaryCrossEntropy, MeanSquareError
from activations import Activations, Sigmoid
from dataset import MNISTDataset

import sklearn
import sklearn.datasets

class MLP():
    def __init__(self, input_features: int, hidden_layers: List[int], 
                 output_features:int, activations: List[Activations], 
                 loss: Losses,
                 learning_rate: float = 0.01):
        self.input_features = input_features
        self.output_features = output_features
        self.dense_layers = []
        self.activations = []
        current_input_features = input_features
        self.loss = BinaryCrossEntropy()
        if loss == Losses.MEAN_SQUARE_ERROR:
            self.loss = MeanSquareError()
        for i, layer in enumerate(hidden_layers + [output_features]):
            self.dense_layers.append(Dense(current_input_features, layer, learning_rate, Intializer.NORMAL))
            activation = activations[i]
            if activation != Activations.SIGMOID:
                raise ValueError("Only Sigmoid activation is supported.")
            self.activations.append(Sigmoid())
            current_input_features = layer
    
    def forward(self, X:np.ndarray):
        output_ = X
        for i, layer in enumerate(self.dense_layers):
            output_ = layer.forward(output_)
            output_ = self.activations[i].forward(output_)
        return output_
        
    def fit(self, X:np.ndarray, Y:np.ndarray, iters:int):
        for i in range(iters):
            y_pred = self.forward(X)
            loss, gradient = self.loss.compute_loss_and_gradient(y_pred, Y)
            if i%1000 == 0:
                print(f"For iter {i}, training loss is {loss}")
            # Backpropagation
            for i in range(len(self.dense_layers) -1 , -1, -1):
                gradient = self.activations[i].backprop(gradient)
                gradient = self.dense_layers[i].backprop(gradient)

if __name__ == "__main__":
    train_X, train_Y = MNISTDataset(n_classes=2).load_data()
    mlp = MLP(input_features=train_X.shape[1], hidden_layers=[128, 128], 
              output_features=train_Y.shape[1], activations=[Activations.SIGMOID, Activations.SIGMOID, Activations.SIGMOID],
              loss=Losses.BINARY_CROSS_ENTROPY, learning_rate=0.01)
    mlp.fit(train_X[:1000], train_Y[:1000], iters=10001)