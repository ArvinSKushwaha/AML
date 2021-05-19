from typing import List
from .core import Tensor


class Optimizer:  # The Optimizer class is an abstract class meant for child classes to extend to suit their needs.
    def __init__(self, parameters: List[Tensor]) -> None:
        self.parameters = parameters

    def zero_grad(self):  # This method will return all gradients to 0
        for n in range(len(self.parameters)):
            self.parameters[n].grad.data.fill(0)

    def step(
        self,
    ):  # The application of the gradients to the parameters must be implemented by child classes.
        raise NotImplementedError()


class SGD(Optimizer):  # The well known Stochastic Gradient Descent
    def __init__(self, parameters: List[Tensor], learning_rate: float = 1e-2) -> None:
        super().__init__(parameters)

        self.lr = learning_rate

    def step(self):
        for n in range(len(self.parameters)):
            assert (
                self.parameters[n].shape
                == self.parameters[n].shape
                == self.parameters[n].grad_free.shape
            )
            self.parameters[n].data -= (
                self.lr * self.parameters[n].grad.data
            )  # Simply scale the gradients by the learning rates before "descending"
            self.parameters[n].grad_free.data = self.parameters[
                n
            ].data  # Ensure that the .grad_free has the same data as our tensor.


class Adam(Optimizer):
    def __init__(
        self,
        parameters: List[Tensor],
        learning_rate: float = 1e-4,
        betas=(0.9, 0.999),
        eps=1e-6,
    ) -> None:  # This is the famous Adam optimizer
        super().__init__(parameters)

        # Hyperparameters
        self.lr = learning_rate
        self.b1, self.b2 = betas
        self.eps = eps

        # Internal parameters
        self.iterations = 0
        self.m = [0 for i in self.parameters]
        self.v = [0 for i in self.parameters]

    def step(self):
        self.iterations += 1
        for n in range(len(self.parameters)):
            assert (
                self.parameters[n].shape
                == self.parameters[n].shape
                == self.parameters[n].grad_free.shape
            )  # Require that the gradients and data have not suddenly changed shape

            # Update internal parameters
            self.m[n] = (
                self.b1 * self.m[n] + (1 - self.b1) * self.parameters[n].grad.data
            )
            self.v[n] = self.b2 * self.v[n] + (1 - self.b2) * (
                self.parameters[n].grad.data ** 2
            )

            m_hat = self.m[n] / (1 - self.b1 ** self.iterations)
            v_hat = self.v[n] / (1 - self.b2 ** self.iterations)

            # Update the model parameters
            self.parameters[n].data -= self.lr * m_hat / ((v_hat ** 0.5) + self.eps)
            self.parameters[n].grad_free.data = self.parameters[n].data

