//
// Created by ubuntu on 2/9/25.
//

#ifndef UTIL_CUH
#define UTIL_CUH

#include <cuda/atomic>
#include <cuda/std/concepts>
#include <cute/tensor.hpp>

#define SHARED_SIZE 16 * 1024U
#define CAST_TO(T, p) static_cast<T*>(static_cast<void*>(p))
#define BYTE_CAST(p) static_cast<cuda::std::byte*>(static_cast<void*>(p))
#define NANO_TO_MICRO (cuda::std::nano::den / cuda::std::micro::den)
#define TO_MB(b) static_cast<double>(b) / (1024.0f*1024.0f)
#define NANO_TO_MICRO (cuda::std::nano::den / cuda::std::micro::den)
#if !defined(CHECK_ERROR_EXIT)
#  define CHECK_ERROR_EXIT(e)                                         \
do {                                                           \
cudaError_t code = (e);                                      \
if (code != cudaSuccess) {                                   \
fprintf(stderr, "<%s:%d> %s:\n    %s: %s\n",               \
__FILE__, __LINE__, #e,                            \
cudaGetErrorName(code), cudaGetErrorString(code)); \
fflush(stderr);                                            \
exit(1);                                                   \
}                                                            \
} while (0)
#endif

#if !defined(CHECK_LAST)
# define CHECK_LAST() CHECK_ERROR_EXIT(cudaPeekAtLastError()); CHECK_ERROR_EXIT(cudaDeviceSynchronize())
#endif

__device__ __forceinline__
bool isLikelyRegister(const void* const& p) {
    return !(__isShared(p) &&
        __isLocal(p) &&
        __isConstant(p) &&
        __isGlobal(p) &&
        __isGridConstant(p));
}

template<unsigned int Arch>
concept SupportedArch = Arch >= 700 && Arch <= 900;

/// A more apropos name would be "static storage" rather than registers.
template<class T>
struct isRegister : cuda::std::false_type {};

template<class T, int N, int Alignment>
struct isRegister<cutlass::AlignedArray<T, N, Alignment>> : cuda::std::true_type {};

template<class T, int N, bool RegisterSized>
struct isRegister<cutlass::Array<T, N, RegisterSized>> : cuda::std::true_type {};

template<class Engine, class Layout>
struct isRegister<cute::Tensor<Engine, Layout>> :
cuda::std::conditional_t<cute::is_rmem_v<cute::Tensor<Engine, Layout>>,
cuda::std::true_type, cuda::std::false_type> {};

template <class T>
constexpr bool isRegisterV = isRegister<T>::value;

template<typename B>
    concept AtomicType = cuda::std::same_as<B, int> || cuda::std::same_as<B, unsigned int>
    || cuda::std::same_as<B, unsigned long long int>;

template<typename B>
concept AtomicCASType = cuda::std::same_as<B, int> || cuda::std::same_as<B, unsigned int>
|| cuda::std::same_as<B, unsigned long long int> || cuda::std::same_as<B, unsigned short int>;

template<cuda::thread_scope scope>
concept AtomicScope = scope == cuda::thread_scope_thread ||
    scope == cuda::thread_scope_block || scope == cuda::thread_scope_device || scope == cuda::thread_scope_system;

template<cuda::thread_scope scope = cuda::thread_scope_device, typename T>
requires AtomicType<T> && AtomicScope<scope>
__device__ __forceinline__
T atomicLoad(T* __restrict__ const& addr){
    if constexpr (scope == cuda::thread_scope_block || scope == cuda::thread_scope_thread) {
        return atomicOr_block(addr, 0U);
    }
    if constexpr (scope == cuda::thread_scope_system) {
        return atomicOr_system(addr, 0U);
    }
    return atomicOr(addr, 0U);
}

template<cuda::thread_scope scope = cuda::thread_scope_device,
    unsigned int bound = cuda::std::numeric_limits<unsigned int>::max()>
    requires(AtomicScope<scope> && bound <= cuda::std::numeric_limits<unsigned int>::max())
    __device__ __forceinline__
    unsigned int atomicIncrement(unsigned int* __restrict__ const& addr) {
    if constexpr (scope == cuda::thread_scope_block || scope == cuda::thread_scope_thread) {
        return atomicInc_block(addr, bound);
    }
    if constexpr (scope == cuda::thread_scope_system) {
        return atomicInc_system(addr, bound);
    }
    return atomicInc(addr, bound);
}
#endif //UTIL_CUH
