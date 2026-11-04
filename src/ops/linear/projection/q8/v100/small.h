#pragma once
#include "core/tensor.h"
#include "core/weight.h"
#include <cuda_runtime_api.h>

namespace ninfer::ops::detail {
void launch_q8_volta_eight_tokens(const Tensor&, const Weight&, Tensor&, cudaStream_t);
}
