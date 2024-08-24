# Simple Dense / Hidden layer

import numpy as np
from enum import Enum

class Intializer(Enum):
    # Initialize layer with zeros
    ZERO = 1
    RANDOM = 2
    NORMAL = 3


class Dense():
    # Simple Dense layer. 
    # Forward pass is defined as Xw + b where X is the input, w the weight matrix and b the bias term.
    # In numpy it will be defined as MatMul(X, w) + b  
    def __init__(self, in_features: int, out_features: int, learning_rate = 0.01, init: Intializer = Intializer.ZERO):
        self.in_features = in_features
        self.out_features = out_features
        self.mat = None
        self.lr = learning_rate
        if init == Intializer.ZERO:
            self.mat = np.zeros(in_features, out_features)
        if init == Intializer.RANDOM:
            self.mat = np.random.rand(in_features, out_features)
        if init == Intializer.NORMAL:
            self.mat = np.random.randn(in_features, out_features)
        # Bias term is always initialized to 0.
        self.bias = np.zeros(out_features)
        # Layer Input. This is needed to compute gradient
        self.input_x = None
    
    def forward(self, X: np.array) -> np.array:
        """
        Forward Pass
        Args:
         X: Input Data
        
        Returns:
         Result of X * mat + bias
        """
        # Forward Pass through the layer, expected ouput shape is (X.shape[0], out_features)
        assert X.shape[1] == self.in_features, f"Num features does not match, expected {self.in_features} got {X.shape[1]}"
        self.input_x = X
        return np.matmul(X, self.mat) + self.bias

    def backprop(self, partial_gradient: np.array):
        """
        Compute Gradients and update weights
        Args:
            partial_gradient: Gradient values from downstream (in forward flow) layers
        
        Returns:
            Partial Gradients for upstream layers

        For Mat's gradient we need to multiply partial_gradient with self.input_x. The aim is to ensure that the shape of 
        the product of partial_gradient and self.input_x matches Mat's shape. 
        """
        assert self.input_x is not None, "self.input_x cannot be None"
        assert partial_gradient.shape[1] == self.out_features, f"Expected partial gradient to be of shape (batch, {self.out_features}), got {partial_gradient.shape}"
        # Derivative wrt Mat.
        # d(Xw + b)/d(w) = X. 
        # So multiply partial_gradient with self.input_x
        delta_mat = 1/self.input_x.shape[0] * np.matmul(self.input_x.T, partial_gradient)
        # The matrix multiplication will give shape in_features, out_features, though it combines the gradient of all samples in the
        # the batch. So need to average it.
        # For the bias term, just need to average out the partial_gradient
        delta_bias = 1/self.input_x.shape[0] * np.sum(partial_gradient, axis=1, keepdims=True)
        self.mat = self.mat - self.lr * delta_mat
        self.bias = self.bias - self.lr * delta_bias
        # Finally we need to also compute the gradient for the the next layer.
        # For this we need to take the partial derivative wrt to the input x. This is because, the
        # current layers input is the previous layer's output, so we need to get this gradient to 
        # to compute the gradient previous layer's mat. Mathematically d(loss)/d(w_prev):
        # d(loss)/d(input_x) * d(input_x)/d(w_prev). So we need to d(loss)/d(input_x).
        # The output shape here would be (batch, in_features)
        return np.matmul(partial_gradient, self.mat.T)
        

        
