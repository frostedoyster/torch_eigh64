import torch
from torch_mkl64 import mkl64

A = torch.rand(2, 2)
print(A)
d, O = mkl64.eigh64(A)

print(d)
print(O)
