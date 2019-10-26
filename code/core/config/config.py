import copyreg
import torch

class PyTorchConfig:
    """ We use this class to set the tensor data type and device to be used
    throughout the whole application. Please let me know if there is a better
    way to write device independent code. """

    def __init__(self, dtype = torch.float32, device = 'cpu'):
        self.dtype = dtype
        self.device = device

    def __eq__(self, other):
        return self.dtype == other.dtype and self.device == other.device

    def __hash__(self):
        return hash((self.dtype, self.device))

# For the multiprocessing package to work, we need all the arguments to be
# pickleable. Currently, objects of type torch.dtype are not pickleable.
# We implement a workaround using copyreg from
# https://github.com/pytorch/text/issues/350.
def unpickle_torch_dtype(torch_dtype: str):
    dtype = torch_dtype.split('.')[1]
    return getattr(torch, dtype)

def pickle_torch_dtype(torch_dtype: torch.dtype):
    return unpickle_torch_dtype, (str(torch_dtype),)

copyreg.pickle(torch.dtype, pickle_torch_dtype)
