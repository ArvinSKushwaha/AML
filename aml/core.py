from numbers import Number
from os import urandom
from typing import Tuple, Union
from functools import reduce
import operator
import numpy as np


rn_state = np.random.RandomState()


def seed_random(x: int):
    global rn_state
    rn_state = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(x)))


# To store the sequence of operations (for automatic differentiation),
# the Ops class holds information on the inputs as well as the method for differentiation.
class Ops:
    def __init__(self, op: str):
        self.op = op
        self.left, self.right = None, None
        self.args = None

    def __repr__(self) -> str:
        if not hasattr(self, "args") or not self.args:
            if not hasattr(self, "right") or not self.right:
                return f"{self.op}({self.left})"
            else:
                return f"{self.op}({self.left}, {self.right})"
        else:
            return f"{self.op}{self.args}"

    def diff(self, diff: "Tensor", value: "Tensor"):
        raise NotImplementedError()


from base64 import (
    b64encode,
)  # To give each tensor an identification string for debugging


class Tensor:
    def __init__(
        self, data: np.ndarray, requires_grad: bool = False, last_op: Ops = None
    ) -> None:  # Tensors store data in numpy arrays, and may or may not keep track of gradients (depends if the tensor is differentiable).
        if not isinstance(data, np.ndarray):
            self.data = np.array(data).astype(
                np.float64
            )  # If inputted data is not an ndarray, make it an ndarray
        else:
            self.data = data.astype(
                np.float64
            )  # If inputted data is an ndarray, convert its entries to 64-bit floats
        self.requires_grad = requires_grad
        if requires_grad:
            self.last_op = last_op  # Only store operations if it is differentiable (otherwise, there's no point in storing it)
        else:
            self.last_op = None

        self.elem_n = self.data.size

        if requires_grad:
            self.grad = Tensor(
                np.zeros_like(self.data), requires_grad=False
            )  # Only store gradients if the tensor is differentiable
            self.grad_free = Tensor(
                self.data
            )  # self.grad_free is a copy of the tensor for which we don't need to concern ourself with its differentiability.
        else:
            self.grad_free = self

        self.ndim = self.data.ndim
        self.size = self.data.shape
        self.shape = self.data.shape
        self.id = b64encode(urandom(2)).decode("utf-8")  # ID comes from random bits.

    # Constructors
    @staticmethod
    def zeros(
        *args: int, requires_grad: bool = False
    ) -> "Tensor":  # Tensors filled with zeros
        return Tensor(np.zeros(args, np.float32), requires_grad)

    @staticmethod
    def ones(
        *args: int, requires_grad: bool = False
    ) -> "Tensor":  # Tensors filled with ones
        return Tensor(np.ones(args, np.float32), requires_grad)

    @staticmethod
    def rand(
        *args: int, requires_grad: bool = False
    ) -> "Tensor":  # Tensors filled with random values from the uniform distribution on [0, 1)
        return Tensor(rn_state.rand(*args).astype(np.float32), requires_grad)

    @staticmethod
    def randn(
        *args: int, requires_grad: bool = False
    ) -> "Tensor":  # Tensors filled with random values from the standard gaussian distribution
        return Tensor(rn_state.randn(*args).astype(np.float32), requires_grad)

    @staticmethod
    def uniform(
        *args: int, low: float = 0.0, high: float = 1.0, requires_grad: bool = False
    ) -> "Tensor":  # Tensors filled with random values from the uniform distribution on [low, high)
        return Tensor(
            rn_state.rand(*args).astype(np.float32) * (high - low) + low, requires_grad
        )

    def backward(
        self,
    ):  # Calculate the gradient of every tensor that this tensor depends on.
        if self.requires_grad:
            self.grad.data.fill(1)  # dx/dx = 1
            if self.last_op:
                self.last_op.diff(self.grad, self.grad_free)  # Chain rule
        else:
            raise Exception(
                "Tensors that don't require grad can't use reverse automatic differentiation"
            )

    def T(
        self,
    ) -> "Tensor":  # Take transpose of tensor, and if it requires gradient, pass the operation to the output tensor
        output = Tensor(self.data.T, self.requires_grad, Transpose(self))
        return output

    def moveaxis(
        self, start_dim: int, end_dim: int
    ) -> "Tensor":  # Moving axes (useful for reshaping)
        output = Tensor(
            np.moveaxis(self.data, start_dim, end_dim),
            self.requires_grad,
            MoveAxis(self),
        )
        return output

    def __add__(
        self, other
    ) -> "Tensor":  # Add tensors together, in these element-wise functions, we'll have a broadcast precursor
        if isinstance(other, Tensor):
            common_shape = np.broadcast_shapes(self.shape, other.shape)
            x, y = (
                broadcast(self, common_shape),
                broadcast(other, common_shape),
            )  # This broadcasting is essential to make sure gradients are accumulated and scatted properly

            output = Tensor(  # If either operand required gradients, the output will too.
                x.data + y.data,
                x.requires_grad or y.requires_grad,
                Add(x, y),  # Create the output tensor and pass the addition operation.
            )
            return output

        return self + Tensor(other)  # If it's not a tensor, make it one.

    def __radd__(self, other) -> "Tensor":
        return self.__add__(other)

    def __sub__(self, other) -> "Tensor":
        if isinstance(other, Tensor):
            common_shape = np.broadcast_shapes(self.shape, other.shape)
            x, y = (
                broadcast(self, common_shape),
                broadcast(other, common_shape),
            )  # Same pattern here, like with addition

            output = Tensor(
                x.data - y.data,
                x.requires_grad or y.requires_grad,
                Sub(x, y),  # Pass the subtraction operation
            )
            return output

        return self - Tensor(other)

    def __rsub__(self, other) -> "Tensor":
        return self.__sub__(other)

    def __neg__(self) -> "Tensor":
        return self * -1  # Negation is nothing but multiplying by -1.

    def __mul__(self, other) -> "Tensor":
        if isinstance(other, Tensor):
            common_shape = np.broadcast_shapes(self.shape, other.shape)
            x, y = broadcast(self, common_shape), broadcast(other, common_shape)

            output = Tensor(
                x.data * y.data,
                x.requires_grad or y.requires_grad,
                Mul(x, y),  # Pass the multiplication operation
            )
            return output

        return self * Tensor(other)

    def __rmul__(self, other) -> "Tensor":
        return self.__mul__(other)

    def __truediv__(self, other) -> "Tensor":
        if isinstance(other, Tensor):
            return self * (
                other ** -1
            )  # Instead of creating a Div operation, I felt it easier to simply multiply by the multiplicative inverse.

        return self / Tensor(other)

    def __rtruediv__(self, other) -> "Tensor":
        return Tensor(other) / self

    def __pow__(
        self, other
    ) -> "Tensor":  # With power, there are some limitations. The exponent must not be differentiable. Future versions of my library will correct for this, however.
        if isinstance(other, Tensor):
            common_shape = np.broadcast_shapes(self.shape, other.shape)
            x, y = broadcast(self, common_shape), broadcast(other, common_shape)

            output = Tensor(
                x.data ** y.data, x.requires_grad or y.requires_grad, Pow(x, y),
            )
            return output

        return self ** Tensor(other)

    def __rpow__(self, other):
        return Tensor(other) ** self

    def __matmul__(
        self, other
    ) -> "Tensor":  # Matrix multiplication, unlike the previous operations, does not have broadcasting yet. Future editions will contain this update.
        if isinstance(other, Tensor):
            output = Tensor(
                self.data @ other.data,
                self.requires_grad or other.requires_grad,
                MatMul(self, other),  # Pass in the matrix multiplication operation.
            )
            return output

        return self @ Tensor(other)

    def __getitem__(
        self, idx
    ) -> "Tensor":  # It was kind of strange to create a differentiable operator for getting and setting elements, but it would be quite important for ReLU operations and whatnot.
        if isinstance(idx, Number):
            idx = (idx,)
        assert (
            len(idx) <= self.ndim
        )  # We musn't have higher dimensional queries than the data itself.
        output = Tensor(
            self.data[idx], self.requires_grad, Index(self, idx)
        )  # Pass in the Indexing operation.
        return output

    def __setitem__(
        self, idx, new_val: "Tensor"
    ) -> "Tensor":  # Unfortunately, I'm still working out the problems in taking gradients of matrices that involve setting items, which is why setting items removes the differentiability of the tensor. Future editions will correct for this, however.
        if isinstance(idx, Number):
            idx = (idx,)
        assert len(idx) <= self.ndim
        if not isinstance(new_val, Tensor):
            new_val = Tensor(new_val)
        self.data[idx] = new_val.data
        # Cannot take gradient of a Tensor which has been altered in-place (for now...)
        if self.requires_grad:
            del self.grad
        self.requires_grad = False

    def __repr__(self) -> str:  # Need to be able to visualize the tensors
        return f"Tensor({self.data}" + (
            f", last_func={self.last_op.op})" if self.last_op else ")"
        )

    def __eq__(
        self, x: "Tensor"
    ) -> "Tensor":  # Compare if tensors are equivalent, element-wise.
        return Tensor(self.data == (x.data if isinstance(x, Tensor) else x))

    def mean(self, axis=None, keepdim: bool = False):  # Take the average over axes
        return mean(self, axis=axis, keepdim=keepdim)

    def sum(self, axis=None, keepdim: bool = False):  # Sum over axes
        return sum(self, axis=axis, keepdim=keepdim)


def exp(x: Tensor) -> Tensor:  # e^x
    if not isinstance(x, Tensor):
        return exp(Tensor(x))
    output = Tensor(
        np.exp(x.data), x.requires_grad, Exp(x)
    )  # Pass the exponentiation operation
    return output


def log(x: Tensor) -> Tensor:  # log(x)
    if not isinstance(x, Tensor):
        return log(Tensor(x))
    output = Tensor(
        np.log(x.data), x.requires_grad, Log(x)
    )  # Pass the logarithmic operation
    return output


def sin(x: Tensor) -> Tensor:  # sin(x)
    if not isinstance(x, Tensor):
        return sin(Tensor(x))
    output = Tensor(np.sin(x.data), x.requires_grad, Sin(x))  # Pass the sine operation
    return output


def cos(x: Tensor) -> Tensor:  # cos(x)
    if not isinstance(x, Tensor):
        return cos(Tensor(x))
    output = Tensor(
        np.cos(x.data), x.requires_grad, Cos(x)
    )  # Pass the cosine operation
    return output


def tan(x: Tensor) -> Tensor:  # tan(x)
    if not isinstance(x, Tensor):
        return tan(Tensor(x))
    return sin(x) / cos(x)  # tan(x) = sin(x) / cos(x)


def mean(
    x: Tensor, axis=None, keepdim: bool = False
) -> Tensor:  # Take the average over axes
    output = Tensor(
        np.mean(x.data, axis=axis, keepdims=keepdim),  # NumPy handles the averaging
        x.requires_grad,
        Mean(
            x, axis=axis, keepdim=keepdim
        ),  # This package handles differentiating over the mean.
    )
    return output


def sum(
    x: Tensor, axis=None, keepdim: bool = False
) -> Tensor:  # Take the sum over axes
    output = Tensor(
        np.sum(x.data, axis=axis, keepdims=keepdim),
        x.requires_grad,
        Sum(x, axis=axis, keepdim=keepdim),  # This package handles summation, too.
    )
    return output


def broadcast(
    x: Tensor, shape=None
) -> Tensor:  # This is the infamous broadcasting function.
    if x.shape == shape:
        return x  # No need for broadcasting is the shape is already what is expected.
    return Tensor(
        np.broadcast_to(x.data, shape), x.requires_grad, Broadcast(x, shape)
    )  # NumPy does the broadcasting itself, this package determines how to invert the broadcasting and which axes to collapse.


class Add(Ops):  # This is Addition operation class
    def __init__(self, left: Tensor, right: Tensor):
        self.op = "+"  # Tells us the operation when debugging
        self.left, self.right = (
            left,
            right,
        )  # Stores our tensors (they're the same size because they've been broadcasted)

    def diff(self, diff: Tensor, value: Tensor):
        if self.left.requires_grad:
            left_ops = self.left.last_op
            self.left.grad += diff  # Derivative of d(x+y)/dx = 1, but we scale by the gradient before that because of chain rule.
            if left_ops:
                left_ops.diff(
                    self.left.grad, self.left.grad_free
                )  # Pass down the gradient to continue chain rule.

        if self.right.requires_grad:
            right_ops = self.right.last_op
            self.right.grad += diff  # Same thing for both sides
            if right_ops:
                right_ops.diff(self.right.grad, self.right.grad_free)


class Sub(
    Ops
):  # Subtracting is just like addition, except the second operand has its gradient negated.
    def __init__(self, left: Tensor, right: Tensor):
        self.op = "-"
        self.left, self.right = left, right

    def diff(self, diff: Tensor, value: Tensor):
        if self.left.requires_grad:
            left_ops = self.left.last_op
            self.left.grad += diff
            if left_ops:
                left_ops.diff(self.left.grad, self.left.grad_free)

        if self.right.requires_grad:
            right_ops = self.right.last_op
            self.right.grad -= diff
            if right_ops:
                right_ops.diff(self.right.grad, self.right.grad_free)


class Mul(
    Ops
):  # Here was just use the product rule. You'll notice the usage of .grad_free here. This is to prevent any of the gradients from becoming differentiable as well.
    def __init__(self, left: Tensor, right: Tensor):
        self.op = "*"
        self.left, self.right = left, right

    def diff(self, diff: Tensor, value: Tensor):

        if self.left.requires_grad:
            left_ops = self.left.last_op
            self.left.grad += diff * self.right.grad_free
            if left_ops:
                left_ops.diff(self.left.grad, self.left.grad_free)

        if self.right.requires_grad:
            right_ops = self.right.last_op
            self.right.grad += diff * self.left.grad_free
            if right_ops:
                right_ops.diff(self.right.grad, self.right.grad_free)


class Pow(
    Ops
):  # Since the exponent must not be differentiable (for now), if the exponent is differentiable, a NotImplementedError will be raised.
    def __init__(self, left: Tensor, right: Tensor):
        self.op = "**"
        self.left, self.right = left, right

    def diff(self, diff: Tensor, value: Tensor):
        if self.right.requires_grad and self.left.requires_grad:
            raise NotImplementedError()

        if self.left.requires_grad:
            left_ops = self.left.last_op
            self.left.grad += diff * (
                self.right.grad_free * self.left.grad_free ** (self.right.grad_free - 1)
            )
            if left_ops:
                left_ops.diff(self.left.grad, self.left.grad_free)

        if self.right.requires_grad:
            right_ops = self.right.last_op
            self.right.grad += diff * value * log(self.left.grad_free)
            if right_ops:
                right_ops.diff(self.right.grad, self.right.grad_free)


class MatMul(
    Ops
):  # Matrix-Multiplication is the most confusing operation to take the gradient of (other than broadcasting).
    # It's quite easy to work out on paper, however, especially with the help of Einstein summation notation.
    def __init__(self, left: Tensor, right: Tensor):
        self.op = "@"
        self.left, self.right = left, right

    def diff(self, diff: Tensor, value: Tensor):
        if self.left.requires_grad:
            left_ops = self.left.last_op
            self.left.grad += (
                diff @ self.right.grad_free.T()
            )  # d(A @ B)/dA = tranpose(B) and d(A @ B)/dB = tranpose(A)
            if left_ops:
                left_ops.diff(self.left.grad, self.left.grad_free)

        if self.right.requires_grad:
            right_ops = self.right.last_op
            self.right.grad += self.left.grad_free.T() @ diff
            if right_ops:
                right_ops.diff(self.right.grad, self.right.grad_free)


class Exp(
    Ops
):  # This is the first operation that takes advantage of value being passed through the diff function.
    def __init__(self, left: Tensor):
        self.op = "exp"
        self.left = left

    def diff(
        self, diff: Tensor, value: Tensor
    ):  # Since d(e^x)/dx = e^x, I can simply use the value as the gradient of the exponential.
        if self.left.requires_grad:
            left_ops = self.left.last_op
            self.left.grad += diff * value
            if left_ops:
                left_ops.diff(self.left.grad, self.left.grad_free)


class Sin(Ops):
    def __init__(self, left: Tensor):
        self.op = "sin"
        self.left = left

    def diff(self, diff: Tensor, value: Tensor):

        if self.left.requires_grad:
            left_ops = self.left.last_op
            self.left.grad += diff * cos(self.left.grad_free)  # d(sin(x))/dx = cos(x)
            if left_ops:
                left_ops.diff(self.left.grad, self.left.grad_free)


class Cos(Ops):
    def __init__(self, left: Tensor):
        self.op = "cos"
        self.left = left

    def diff(self, diff: Tensor, value: Tensor):
        if self.left.requires_grad:
            left_ops = self.left.last_op
            self.left.grad += diff * -sin(self.left.grad_free)  # d(cos(x))/dx = -sin(x)
            if left_ops:
                left_ops.diff(self.left.grad, self.left.grad_free)


class Log(Ops):
    def __init__(self, left: Tensor):
        self.op = "log"
        self.left = left

    def diff(self, diff: Tensor, value: Tensor):
        if self.left.requires_grad:
            left_ops = self.left.last_op
            self.left.grad += diff / self.left.grad_free  # d(log(x))/dx = 1/x
            if left_ops:
                left_ops.diff(self.left.grad, self.left.grad_free)


class Transpose(Ops):
    def __init__(self, left: Tensor):
        self.op = "T"
        self.left = left

    def diff(self, diff: Tensor, value: Tensor):
        if self.left.requires_grad:
            left_ops = self.left.last_op
            self.left.grad += (
                diff.T()
            )  # For transposing, I just transpose again (inverting the tranposition), to add the gradient back to the original tensor.
            if left_ops:
                left_ops.diff(self.left.grad, self.left.grad_free)


class MoveAxis(Ops):
    def __init__(self, left: Tensor, start_axis: int, end_axis: int):
        self.op = "T"
        self.left = left
        self.start, self.end = start_axis, end_axis

    def diff(self, diff: Tensor, value: Tensor):

        if self.left.requires_grad:
            left_ops = self.left.last_op
            self.left.grad += diff.moveaxis(
                self.end, self.start
            )  # Like with transpose, I invert the transposition
            if left_ops:
                left_ops.diff(self.left.grad, self.left.grad_free)


class Mean(Ops):
    def __init__(
        self, left: Tensor, axis: Union[int, Tuple[int]] = None, keepdim: bool = False
    ):
        self.op = "mean"
        self.left = left
        self.axis = (axis,) if not isinstance(axis, tuple) and axis != None else axis
        self.keepdim = keepdim

    def diff(self, diff: Tensor, value: Tensor):
        if self.left.requires_grad:
            left_ops = self.left.last_op
            if self.axis != None:
                self.left.grad += (
                    Tensor(np.expand_dims(diff.data, axis=self.axis))
                    if not self.keepdim
                    else diff
                ) / reduce(
                    operator.mul, [self.left.shape[i] for i in self.axis]
                )  # This section takes the gradients and scatters them over all the axes, whilst inversely scaling the gradients by the number of elements that had been averaged over.
            else:
                self.left.grad += diff / self.left.elem_n
            if left_ops:
                left_ops.diff(self.left.grad, self.left.grad_free)


class Sum(Ops):
    def __init__(
        self, left: Tensor, axis: Union[int, Tuple[int]] = None, keepdim: bool = False
    ):
        self.op = "sum"
        self.left = left
        self.axis = axis
        self.keepdim = keepdim

    def diff(self, diff: Tensor, value: Tensor):
        if self.left.requires_grad:
            left_ops = self.left.last_op
            if self.axis != None and not self.keepdim:
                self.left.grad += Tensor(
                    np.expand_dims(diff.data, axis=self.axis)
                )  # Similar to mean, except no scaling, as the output does not depend on the number of collapsed elements.
            else:
                self.left.grad += diff
            if left_ops:
                left_ops.diff(self.left.grad, self.left.grad_free)


class Index(Ops):
    def __init__(self, left: Tensor, indices: Union[int, slice]):
        self.op = "index"
        self.left = left
        self.indices = indices

    def __repr__(self) -> str:
        return f"Index({self.left}){self.indices}"

    def diff(self, diff: Tensor, value: Tensor):
        if self.left.requires_grad:
            left_ops = self.left.last_op
            self.left.grad[self.indices] += (
                self.left.grad[self.indices] + diff
            )  # To deal with indexing, the gradients are simply returned to the location from which the values had been extracted.
            if left_ops:
                left_ops.diff(self.left.grad, self.left.grad_free)


class Broadcast(Ops):
    # For broadcasting, the inverse of the broadcasting is found, which is a list of integers represents which axes need to be collapse and over which axes the gradient should be collected.
    def __init__(self, left: Tensor, shape: Tuple[int]):
        self.op = "broadcast"
        self.left = left
        self.shape = shape

        ls = list(left.shape)
        rs = list(self.shape)
        rs_ = rs[len(rs) - len(ls) :]
        self.bools = [1] * (len(rs) - len(ls)) + [
            1 if i != j else 0 for i, j in zip(ls, rs_)
        ]
        self.indices = tuple([i for i, e in enumerate(self.bools) if e])

        self.ls, self.rs, self.rs_ = ls, rs, rs_

    def __repr__(self) -> str:
        return f"Index({self.left}){self.indices}"

    def diff(self, diff: Tensor, value: Tensor):
        if self.left.requires_grad:
            left_ops = self.left.last_op
            self.left.grad += Tensor(
                np.sum(diff.data, axis=self.indices).reshape(
                    self.left.grad.shape
                )  # The gradient over all those axes are collected and then added to the tensor's gradient. In a future update, the non-prepended axes' dimensions will be kept, to prevent the need to reshape.
            )
            if left_ops:
                left_ops.diff(self.left.grad, self.left.grad_free)


# Helper functions to initialize tensors.
def tensor(data, requires_grad=False):
    return Tensor(data, requires_grad=requires_grad)


def grad_tensor(data):
    return tensor(data, requires_grad=True)


def zeros(*args, requires_grad=False):
    return Tensor.zeros(*args, requires_grad=requires_grad)


def ones(*args, requires_grad=False):
    return Tensor.ones(*args, requires_grad=requires_grad)


def rand(*args, requires_grad=False):
    return Tensor.rand(*args, requires_grad=requires_grad)


def randn(*args, requires_grad=False):
    return Tensor.randn(*args, requires_grad=requires_grad)


# The argmax function for dealing with one-hot encoding.
def argmax(x: Tensor, axis=None):
    return Tensor(np.argmax(x.data, axis=axis))
