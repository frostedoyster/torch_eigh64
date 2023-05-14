import torch
import torch_eigh64

A = torch.random.rand(5, 5)
d, O = torch_eigh64.eigh64(A)

print(d, O)
