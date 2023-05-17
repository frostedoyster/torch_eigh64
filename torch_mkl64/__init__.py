import torch
torch.set_default_dtype(torch.float64)
from torch.utils.cpp_extension import load
import os

os.environ["CXX"] = "g++"  # Force g++ as it is the compiler that compiles the pre-built wheels for pytorch
MKLROOT = os.environ["MKLROOT"]
this_path = os.path.dirname(os.path.abspath(__file__))

mkl64 = load(
    name='mkl64', 
    sources=[os.path.join(this_path, 'mkl64.cc')], 
    extra_cflags=['-DMKL_ILP64', '-m64', f'-I"{MKLROOT}/include"'],
    extra_ldflags=[f'-L{MKLROOT}/lib/intel64', '-Wl,--no-as-needed', '-lmkl_intel_ilp64', '-lmkl_intel_thread', '-lmkl_core', '-liomp5', '-lpthread', '-lm', '-ldl']
)
