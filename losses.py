from typing import Any, Tuple
from enum import Enum
import numpy as np

class Losses(Enum):
    BINARY_CROSS_ENTROPY = 1
    MEAN_SQUARE_ERROR = 2

class Loss():
    def compute_loss_and_gradient(self, y_pred:np.ndarray, y_gt:np.ndarray) -> Tuple[np.floating[Any], np.ndarray]: # type: ignore
        """
        Compute the Loss
        Args:
            y_pred: Predicted Output
            y_gt: Ground Truth Output
        Returns the loss value and the corresponding gradient.
        """
        pass

class BinaryCrossEntropy(Loss):
    def compute_loss_and_gradient(self, y_pred:np.ndarray, y_gt:np.ndarray) -> Tuple[np.floating[Any], np.ndarray]:
        loss = -np.sum(y_gt*np.log(y_pred) + (1-y_gt)*np.log(1-y_pred))/y_pred.shape[0]
        gradient = (y_pred - y_gt) / (y_pred * (1- y_pred))
        return (loss, gradient)

class MeanSquareError(Loss):
    def compute_loss_and_gradient(self, y_pred:np.ndarray, y_gt:np.ndarray) -> Tuple[np.floating[Any], np.ndarray]:
        loss = np.sum(np.square(y_pred - y_gt))/y_pred.shape[0]
        gradient = 2 * (y_pred - y_gt)
        return (loss, gradient)