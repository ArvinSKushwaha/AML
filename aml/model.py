from typing import List
from .core import Tensor


# This is the module class. Almost all building blocks are constructed from these.
class Module:
    def __init__(self) -> None:
        self.parameters: List[Tensor] = []  # This stores all the parameters

    def register(self, x: Tensor) -> Tensor:  # When we register a tensor
        x = Tensor(x.data, requires_grad=True)  # We make it differentiable
        self.parameters.append(x)  # We add it to the parameters
        return x

    def forward(
        self, *args: Tensor
    ) -> Tensor:  # The forward pass must be implemented by the subclasses
        return NotImplemented

    def __call__(
        self, *args: Tensor
    ) -> Tensor:  # When the module is called, the .forward method is called
        return self.forward(*args)


class Sequential(
    Module
):  # The "Sequential" module is a shortcut for chaining as series of modules
    def __init__(self, *args: Module) -> None:  # args is a list of modules
        super().__init__()
        self.modules = args
        for module in args:
            self.parameters += (
                module.parameters
            )  # All the parameters from each of the modules are added

    def forward(self, *args: Tensor) -> Tensor:
        x = self.modules[0](*args)
        for module in self.modules[
            1:
        ]:  # Each of the modules is called, one after the other.
            x = module(x)

        return x  # Return the output
