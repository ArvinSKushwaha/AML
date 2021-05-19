from .core import Tensor
from random import shuffle as shuffle_


# This class lets us minibatch our data
class TensorSample:
    def __init__(
        self, *tensors: Tensor, batch_size=4, shuffle=True
    ):  # Pass in a collection of tensors, as well as a batch_size, and if we need to shuffle.
        self.tensors = tensors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.length = min(
            [x.shape[0] for x in tensors]
        )  # Get the minimum length to prevent from indexing past the length of the tensor
        self.indices = list(range(self.length))  # Store indices
        if self.shuffle:
            shuffle_(self.indices)  # And shuffle them if needed
        self.index = 0  # Start indexing from 0

    def __len__(self):
        return (
            self.length + self.batch_size - 1
        ) // self.batch_size  # This is the number of minibatches we'll have

    def __iter__(self):
        if self.shuffle:
            shuffle_(self.indices)  # Reshuffle the indices
        self.index = 0  # Restart from 0
        return self

    def __next__(self):
        if self.index < self.length:
            output = tuple(
                [
                    tensor[
                        (
                            self.indices[
                                self.index : min(
                                    self.index + self.batch_size, self.length
                                )
                            ],
                        )
                    ]  # Get a subset of the data for each tensor
                    for tensor in self.tensors
                ]
            )
            self.index += self.batch_size  # Advance the index
            return output

        raise StopIteration  # If the end has been reached, stop iterating.
