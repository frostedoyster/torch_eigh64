import torch
torch.set_default_dtype(torch.float64)
from torch_mkl64 import mkl64

A = torch.rand(5, 5)
A = A + A.T
d, O = mkl64.dsyev(A)
assert torch.allclose(A, O @ torch.diag(d) @ O.T)

A = torch.rand(5, 5)
b = torch.rand(5)
x = mkl64.dgesv(A, b)
assert torch.allclose(b, A @ x)

A = torch.rand(5, 5)
A = A + A.T
b = torch.rand(5)
x = mkl64.dsysv(A, b)
assert torch.allclose(b, A @ x)
