import numpy as np
from typing import List
from dense_layer import Dense, Intializer
from losses import Losses, BinaryCrossEntropy, MeanSquareError
from activations import Activations, Sigmoid

import sklearn
import sklearn.datasets

class MLP():
    def __init__(self, input_features: int, hidden_layers: List[int], 
                 output_features:int, activation: Activations, 
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
        for layer in hidden_layers + [output_features]:
            self.dense_layers.append(Dense(current_input_features, layer, learning_rate, Intializer.NORMAL))
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

def load_data():
    N = 200
    gq = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.7,
                                                  n_samples=N, n_features=2,
                                                  n_classes=2, shuffle=True,
                                                  random_state=None)
    return gq

if __name__ == "__main__":
    print("Main")
    X, Y = load_data()
    mlp = MLP(input_features=2, hidden_layers=[10], 
              output_features=1, activation=Activations.SIGMOID, 
              loss=Losses.BINARY_CROSS_ENTROPY, learning_rate=0.1)
    # train model on sample dataset
    mlp.fit(X, np.expand_dims(Y,axis=1), iters=5000)