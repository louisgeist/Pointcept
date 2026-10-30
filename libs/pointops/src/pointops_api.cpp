#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "knn_query/knn_query_cuda_kernel.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knn_query_cuda", &knn_query_cuda, "knn_query_cuda");
}
