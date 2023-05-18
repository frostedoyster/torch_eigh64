import torch
torch.set_default_dtype(torch.float64)
from torch_mkl64 import mkl64

A = torch.rand(5, 5)
d, O = mkl64.dsyev(A)
print(A)
print(O @ torch.diag(d) @ O.T)
assert torch.allclose(A, O @ torch.diag(d) @ O.T)

A = torch.rand(5, 5)
b = torch.rand(5)
x = mkl64.dgesv(A, b)
assert torch.allclose(b, A @ x)
