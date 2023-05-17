#include <torch/extension.h>
#include <iostream>
#include <stdexcept>
#include "mkl.h"

std::vector<torch::Tensor> eigh64(torch::Tensor A) {

    // A few checks:
    if (A.sizes().size() != 2) throw std::runtime_error("not two-dimensional");
    if (A.sizes()[0] != A.sizes()[1]) throw std::runtime_error("not square");
    if (A.dtype() != torch::kFloat64) throw std::runtime_error("not float64");
    if (!A.is_contiguous()) throw std::runtime_error("not contiguous");

    std::size_t n = A.sizes()[0];
    torch::Tensor O = A.detach().clone();
    torch::Tensor d = torch::empty({A.sizes()[0]}, torch::TensorOptions().dtype(torch::kFloat64));

    LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, O.data_ptr<double>(), n, d.data_ptr<double>());

    return std::vector<torch::Tensor>{d, O};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("eigh64", &eigh64, "torch.linalg.eigh64 for large matrices");
}
