// Thin portability layer for the CUDA implementation.
//
// Compiled by nvcc (__CUDACC__ defined) it maps onto the CUDA runtime and CUB.
// Compiled by a plain C++ compiler it provides a serial emulation: every
// kernel launch runs as a single thread, which is exact for the grid-stride
// kernels used here (they use atomics but never __syncthreads or shared
// memory). The emulation exists only to validate the GPU algorithm on
// machines without an NVIDIA GPU; its timings are meaningless.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#ifdef __CUDACC__
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        cudaError_t err_ = (call);                                                             \
        if (err_ != cudaSuccess) {                                                             \
            std::fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(err_), __FILE__, \
                         __LINE__);                                                            \
            std::exit(1);                                                                      \
        }                                                                                      \
    } while (0)

#else  // ---------------------------------------------------------- emulation

#define __global__
#define __device__
#define __host__
#define CUDA_CHECK(call) (call)

struct dim3 {
    unsigned x = 1, y = 1, z = 1;
};
inline dim3 threadIdx, blockIdx, blockDim, gridDim;

inline int atomicAdd(int* addr, int v) {
    int old = *addr;
    *addr += v;
    return old;
}
inline unsigned atomicAdd(unsigned* addr, unsigned v) {
    unsigned old = *addr;
    *addr += v;
    return old;
}
#endif

namespace gpu {

constexpr unsigned kBlock = 256;

inline unsigned grid_for(size_t n) {
    size_t g = (n + kBlock - 1) / kBlock;
    return static_cast<unsigned>(std::max<size_t>(1, std::min<size_t>(g, 65535)));
}

#ifdef __CUDACC__
template <typename... KArgs, typename... Args>
void launch(void (*kernel)(KArgs...), size_t n, Args... args) {
    if (n == 0) return;
    kernel<<<grid_for(n), kBlock>>>(args...);
    CUDA_CHECK(cudaGetLastError());
}
#else
template <typename... KArgs, typename... Args>
void launch(void (*kernel)(KArgs...), size_t n, Args... args) {
    if (n == 0) return;
    threadIdx.x = 0;
    blockIdx.x = 0;
    blockDim.x = 1;
    gridDim.x = 1;
    kernel(args...);
}
#endif

// ---------------------------------------------------------------- memory

inline size_t g_current_bytes = 0;
inline size_t g_peak_bytes = 0;

inline void* raw_alloc(size_t bytes) {
    void* p = nullptr;
    bytes = std::max<size_t>(bytes, 1);
#ifdef __CUDACC__
    CUDA_CHECK(cudaMalloc(&p, bytes));
#else
    p = std::malloc(bytes);
    if (!p) {
        std::fprintf(stderr, "out of memory allocating %zu bytes\n", bytes);
        std::exit(1);
    }
#endif
    g_current_bytes += bytes;
    g_peak_bytes = std::max(g_peak_bytes, g_current_bytes);
    return p;
}

inline void raw_free(void* p, size_t bytes) {
    if (!p) return;
#ifdef __CUDACC__
    cudaFree(p);
#else
    std::free(p);
#endif
    g_current_bytes -= std::max<size_t>(bytes, 1);
}

inline void copy_h2d(void* dst, const void* src, size_t bytes) {
    if (!bytes) return;
#ifdef __CUDACC__
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
#else
    std::memcpy(dst, src, bytes);
#endif
}

inline void copy_d2h(void* dst, const void* src, size_t bytes) {
    if (!bytes) return;
#ifdef __CUDACC__
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
#else
    std::memcpy(dst, src, bytes);
#endif
}

inline void fill_zero(void* dst, size_t bytes) {
    if (!bytes) return;
#ifdef __CUDACC__
    CUDA_CHECK(cudaMemset(dst, 0, bytes));
#else
    std::memset(dst, 0, bytes);
#endif
}

inline void synchronize() {
#ifdef __CUDACC__
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

inline std::string device_name() {
#ifdef __CUDACC__
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s (%d SMs, %.1f GB, CC %d.%d)", prop.name, prop.multiProcessorCount,
                  prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0), prop.major, prop.minor);
    return buf;
#else
    return "CPU emulation of CUDA kernels (serial, for validation only)";
#endif
}

// Grow-only device buffer. reserve() does not preserve contents.
template <typename T>
class DevBuf {
public:
    DevBuf() = default;
    explicit DevBuf(size_t n) { reserve(n); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    ~DevBuf() { release(); }

    T* get() const { return ptr_; }
    size_t capacity() const { return cap_; }
    void swap(DevBuf& o) {
        std::swap(ptr_, o.ptr_);
        std::swap(cap_, o.cap_);
    }

    void reserve(size_t n) {
        if (n <= cap_) return;
        size_t c = std::max(n, cap_ + cap_ / 2);
        release();
        ptr_ = static_cast<T*>(raw_alloc(c * sizeof(T)));
        cap_ = c;
    }
    void release() {
        raw_free(ptr_, cap_ * sizeof(T));
        ptr_ = nullptr;
        cap_ = 0;
    }
    void upload(const T* src, size_t n) {
        reserve(n);
        copy_h2d(ptr_, src, n * sizeof(T));
    }
    void upload(const std::vector<T>& v) { upload(v.data(), v.size()); }
    void download(T* dst, size_t n) const { copy_d2h(dst, ptr_, n * sizeof(T)); }
    void download(std::vector<T>& v, size_t n) const {
        v.resize(n);
        download(v.data(), n);
    }
    T at(size_t i) const {
        T v;
        copy_d2h(&v, ptr_ + i, sizeof(T));
        return v;
    }
    void zero(size_t n) {
        reserve(n);
        fill_zero(ptr_, n * sizeof(T));
    }

private:
    T* ptr_ = nullptr;
    size_t cap_ = 0;
};

// ------------------------------------------------------- device primitives

struct SumOp {
    __host__ __device__ int operator()(const int& a, const int& b) const { return a + b; }
};

// Wrappers over CUB device-wide primitives (or std:: equivalents in emulation).
// One growable scratch buffer is reused across calls.
class Primitives {
public:
    void exclusive_sum(const uint32_t* in, uint32_t* out, int n) {
        if (n <= 0) return;
#ifdef __CUDACC__
        size_t bytes = 0;
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, bytes, in, out, n));
        temp_.reserve(bytes);
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(temp_.get(), bytes, in, out, n));
#else
        uint32_t acc = 0;
        for (int i = 0; i < n; ++i) {
            uint32_t v = in[i];
            out[i] = acc;
            acc += v;
        }
#endif
    }

    // Sorts (key, value) pairs by the low end_bit bits of the key.
    void sort_pairs(const uint64_t* keys_in, uint64_t* keys_out, const int32_t* vals_in, int32_t* vals_out, int n,
                    int end_bit) {
        if (n <= 0) return;
#ifdef __CUDACC__
        size_t bytes = 0;
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys_in, keys_out, vals_in, vals_out, n, 0,
                                                   end_bit));
        temp_.reserve(bytes);
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(temp_.get(), bytes, keys_in, keys_out, vals_in, vals_out, n, 0,
                                                   end_bit));
#else
        (void)end_bit;
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return keys_in[a] < keys_in[b]; });
        std::vector<uint64_t> k(n);
        std::vector<int32_t> v(n);
        for (int i = 0; i < n; ++i) {
            k[i] = keys_in[idx[i]];
            v[i] = vals_in[idx[i]];
        }
        std::copy(k.begin(), k.end(), keys_out);
        std::copy(v.begin(), v.end(), vals_out);
#endif
    }

    // Sums values of consecutive equal keys. Returns the number of runs.
    int reduce_by_key_sum(const uint64_t* keys_in, uint64_t* keys_out, const int32_t* vals_in, int32_t* vals_out,
                          int n) {
        if (n <= 0) return 0;
#ifdef __CUDACC__
        num_runs_.reserve(1);
        size_t bytes = 0;
        CUDA_CHECK(cub::DeviceReduce::ReduceByKey(nullptr, bytes, keys_in, keys_out, vals_in, vals_out,
                                                  num_runs_.get(), SumOp(), n));
        temp_.reserve(bytes);
        CUDA_CHECK(cub::DeviceReduce::ReduceByKey(temp_.get(), bytes, keys_in, keys_out, vals_in, vals_out,
                                                  num_runs_.get(), SumOp(), n));
        return num_runs_.at(0);
#else
        int r = -1;
        for (int i = 0; i < n; ++i) {
            if (r < 0 || keys_out[r] != keys_in[i]) {
                keys_out[++r] = keys_in[i];
                vals_out[r] = 0;
            }
            vals_out[r] += vals_in[i];
        }
        return r + 1;
#endif
    }

private:
    DevBuf<char> temp_;
#ifdef __CUDACC__
    DevBuf<int> num_runs_;
#endif
};

}  // namespace gpu
