from .core import Tensor, exp, mean, sum, sin, cos, tan
from .model import Module
import numpy as np


# Mean-Square Error
def mse_loss(x: Tensor, y: Tensor) -> Tensor:
    return mean((x - y) ** 2)


# Logistic Function (the exp(-x) is used to prevent floating-point overflow)
def sigmoid(x: Tensor) -> Tensor:
    return 1 / (1 + exp(-x))


# Softmax Function
def softmax(x: Tensor, axis=-1) -> Tensor:
    x = exp(x - np.nanmax(x.data, axis=axis, keepdims=True))
    return x / sum(x, axis=axis, keepdim=True)


# SiLU (also known as the Swish function, it is an activation function first introduced in `Gaussian Error Linear Units (GELUs)`)
def silu(x: Tensor) -> Tensor:
    return sigmoid(x) * x


# The one_hot encoding to convert our integer labels to vectors i.e. 7 -> [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
def one_hot(x: Tensor, num_classes: int) -> Tensor:
    return Tensor(np.eye(num_classes)[x.data.astype(int)])


# A module wrapper for the silu function
class SiLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return silu(x)


# A module wrapper for the softmax function
class Softmax(Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return softmax(x, axis=self.axis)


# A module wrapper for the sigmoid function
class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(x)
