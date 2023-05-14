from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension
import os

os.environ["CXX"] = "icpc"

extension = cpp_extension.CppExtension(
    'eigh64', 
    ['torch_eigh64/eigh64.cc'], 
    extra_compile_args=['-w', '-lmkl'],
    extra_ldflags=['-L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl']
)

ext_modules = [extension]

setup(
    name='torch_eigh64',
    packages = find_packages(),
    ext_modules = ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
