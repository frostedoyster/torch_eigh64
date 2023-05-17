from setuptools import setup, find_packages

setup(
    name='torch_mkl64',
    packages = find_packages(),
    include_package_data=True,
    package_data={'': ['mkl64.cc']},
)
