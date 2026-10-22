// RMSNorm in CUDA C++, for comparison with the Triton kernel in kernels.py.
//
//   y = x / sqrt(mean(x^2) + eps) * w
//
// One thread block normalizes one row. Each thread loads 16 bytes at a time (8 fp16 or
// bf16 values), squares and sums them in fp32, and the block adds up the per-thread sums
// with warp shuffles. A second pass reads the row again, which on an A100 mostly hits
// the L1 and L2 caches, and writes the output.
//
// Built at runtime by cuda_rmsnorm.py with torch.utils.cpp_extension.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cstdint>

namespace {

// VEC values of scalar_t packed so one load or store moves them all.
template <typename scalar_t, int VEC>
struct alignas(sizeof(scalar_t) * VEC) Packed {
  scalar_t v[VEC];
};

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffff, v, offset);
  return v;
}

// Sum v over every thread in the block. Every thread gets the result.
__device__ float block_sum(float v) {
  __shared__ float warp_totals[32];
  __shared__ float total;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  v = warp_sum(v);
  if (lane == 0) warp_totals[warp] = v;
  __syncthreads();

  if (warp == 0) {
    const int warps = (blockDim.x + 31) >> 5;
    v = lane < warps ? warp_totals[lane] : 0.0f;
    v = warp_sum(v);
    if (lane == 0) total = v;
  }
  __syncthreads();
  return total;
}

template <typename scalar_t, int VEC>
__global__ void rmsnorm_kernel(const scalar_t* __restrict__ x, const scalar_t* __restrict__ w,
                               scalar_t* __restrict__ y, int n, float eps) {
  using P = Packed<scalar_t, VEC>;
  const int64_t row = blockIdx.x;
  const P* xp = reinterpret_cast<const P*>(x + row * n);
  const P* wp = reinterpret_cast<const P*>(w);
  P* yp = reinterpret_cast<P*>(y + row * n);
  const int packs = n / VEC;

  float sum_sq = 0.0f;
  for (int i = threadIdx.x; i < packs; i += blockDim.x) {
    const P a = xp[i];
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      const float f = static_cast<float>(a.v[j]);
      sum_sq += f * f;
    }
  }
  const float rstd = rsqrtf(block_sum(sum_sq) / n + eps);

  for (int i = threadIdx.x; i < packs; i += blockDim.x) {
    const P a = xp[i];
    const P b = wp[i];
    P out;
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      out.v[j] = static_cast<scalar_t>(static_cast<float>(a.v[j]) * rstd * static_cast<float>(b.v[j]));
    }
    yp[i] = out;
  }
}

bool aligned16(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }

}  // namespace

torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor w, double eps) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda(), "x and w must be on the GPU");
  TORCH_CHECK(x.scalar_type() == w.scalar_type(), "x and w must have the same dtype");

  const auto x2 = x.reshape({-1, x.size(-1)}).contiguous();
  const auto wc = w.contiguous();
  const int64_t rows = x2.size(0);
  const int64_t n = x2.size(1);
  TORCH_CHECK(wc.numel() == n, "w must have ", n, " elements");
  auto y = torch::empty_like(x2);
  if (rows == 0) return y.reshape(x.sizes());

  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                  x2.scalar_type(), "rmsnorm", [&] {
    constexpr int VEC = 16 / sizeof(scalar_t);
    const bool vectorize = n % VEC == 0 && aligned16(x2.data_ptr()) &&
                           aligned16(wc.data_ptr()) && aligned16(y.data_ptr());
    const int64_t packs = vectorize ? n / VEC : n;
    const int threads = static_cast<int>(std::min<int64_t>(1024, (packs + 31) / 32 * 32));
    const auto* xd = x2.data_ptr<scalar_t>();
    const auto* wd = wc.data_ptr<scalar_t>();
    auto* yd = y.data_ptr<scalar_t>();
    if (vectorize) {
      rmsnorm_kernel<scalar_t, VEC><<<rows, threads, 0, stream>>>(xd, wd, yd, n, static_cast<float>(eps));
    } else {
      rmsnorm_kernel<scalar_t, 1><<<rows, threads, 0, stream>>>(xd, wd, yd, n, static_cast<float>(eps));
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y.reshape(x.sizes());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm", &rmsnorm, "RMSNorm (CUDA)", py::arg("x"), py::arg("w"), py::arg("eps") = 1e-6);
}
