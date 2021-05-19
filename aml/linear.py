from typing import Iterable
from .model import Module
from .core import Tensor


# Linear module == Fully-Connected layer
class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.weights = self.register(
            self.initialize_weights()
        )  # Registering can be seen in the model.py file
        if self.bias:
            self.biases = self.register(
                self.initialize_biases()
            )  # Registering tells the module to add the tensor to the parameters as well as making it differentiable. In future versions, contexts will be included to prevent unnecessary differentiability.

    def initialize_weights(self) -> Tensor:
        stddev = (6 / (self.in_features + self.out_features)) ** 0.5
        return Tensor.uniform(
            self.out_features, self.in_features, low=-stddev, high=stddev
        )  # Using He-intialization to prevent any overflows

    def initialize_biases(self) -> Tensor:
        stddev = (6 / (self.in_features + self.out_features)) ** 0.5
        return Tensor.uniform(
            self.out_features, low=-stddev, high=stddev
        )  # Using He-intialization to prevent any overflows

    # The forward function defines the output of the module
    def forward(self, x: Tensor) -> Tensor:
        if not self.bias:
            return x @ self.weights.T()  # If no bias, just matrix multiply
        else:
            return (
                x @ self.weights.T()
            ) + self.biases  # If bias, add in the biases too.
