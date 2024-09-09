import numpy as np
from enum import Enum

class Activations(Enum):
    SIGMOID = 1

class Sigmoid():
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.result = 1/(1+np.exp(-x))
        return self.result
    
    def backprop(self, partial_gradient: np.ndarray):
        # Derivative of sigmoid = sigmoid(x)*(1-sigmoid(x))
        # Performing element wise multiplication with sigmoid's derivative
        return partial_gradient * (self.result * (1-self.result))