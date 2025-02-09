//
// Created by ubuntu on 2/9/25.
//

#ifndef GEMM_CUH
#define GEMM_CUH

#include <cublasdx.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/collective/builders/sm90_common.inl>
#include <cutlass/gemm/collective/collective_builder.hpp>

#define MIN_ARCH 700U
#define THREADS 128U
#define BLOCK_M 128U
#define BLOCK_N 64U
#define BLOCK_K_HALF 16U
#define BLOCK_K_FULL 8U
#define MAX_REGS (BLOCK_M * BLOCK_N) / THREADS
#define PIPELINE_STAGES 2U

template<typename V>
        concept TensorValueType = cuda::std::is_same_v<V, cute::half_t> ||
            cuda::std::is_same_v<V, cute::bfloat16_t> ||
            cuda::std::is_same_v<V, cute::tfloat32_t> ||
            cuda::std::is_same_v<V, float> ||
            cuda::std::is_same_v<V, cute::float_e4m3_t> ||
            cuda::std::is_same_v<V, cute::float_e5m2_t>;

template<typename T>
concept Tensor = cute::is_tensor<T>::value && TensorValueType<typename T::value_type>;

template<typename M>
concept Matrix = requires(M m){
    requires Tensor<M> && rank(m) == 2;
};

template<typename S>
struct ToCute {
    using T = S;
    static_assert(TensorValueType<T>);
};
template<>
struct ToCute<__half> {
    using T = cute::half_t;
};
template<>
struct ToCute<__nv_bfloat16> {
    using T = cute::bfloat16_t;
};
template<>
struct ToCute<__nv_fp8_e4m3> {
    using T = cute::float_e4m3_t;
};
template<>
struct ToCute<__nv_fp8_e5m2> {
    using T = cute::float_e5m2_t;
};

template<typename S>
requires(TensorValueType<S>)
struct ToCDx {
    using T = S;
};
template<>
struct ToCDx<cute::half_t> {
    using T = __half;
};
template<>
struct ToCDx<cute::bfloat16_t> {
    using T = __nv_bfloat16;
};
template<>
struct ToCDx<cute::float_e4m3_t> {
    using T = __nv_fp8_e4m3;
};
template<>
struct ToCDx<cute::float_e5m2_t> {
    using T = __nv_fp8_e5m2;
};

template<
    typename ElementA,
    typename ElementB,
    typename ElementC,
    typename ActivationOp
>
struct BlockMM<900, ElementA, ElementB, ElementC, ActivationOp> {
    static_assert(BLOCK_M == THREADS);
    static_assert(BLOCK_M == 128);
    static_assert(BLOCK_N == 64, "64 is a very good value for N, change it back!");
    using GEMM = decltype(cublasdx::Size<BLOCK_M, BLOCK_N, sizeof(ElementA) == 4 ? BLOCK_K_FULL : BLOCK_K_HALF>()
                          + cublasdx::Precision<typename ToCDx<ElementA>::T,
                          typename ToCDx<ElementB>::T, typename ToCDx<ElementC>::T>()
                          + cublasdx::Type<cublasdx::type::real>()
                          + cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major, cublasdx::row_major>()
                          + cublasdx::Function<cublasdx::function::MM>()
                          + cublasdx::SM<900>()
                          + cublasdx::Block()
                          + cublasdx::BlockDim<THREADS>());
    using MatrixAType = ElementA;
    using MatrixBType = ElementB;
    using MatrixCType = ElementC;
    using MatrixDType = ElementA;
    using BlockTiler = cute::Shape<cute::Int<cublasdx::size_of<GEMM>::m>,
                                    cute::Int<cublasdx::size_of<GEMM>::n>,
                                    cute::Int<cublasdx::size_of<GEMM>::k>>;
    using MMBuilder = cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        ElementA,
        cutlass::layout::RowMajor, 16 / sizeof(ElementA),
        ElementB,
        cutlass::layout::RowMajor, 16 / sizeof(ElementB),
        ElementC, cute::Shape<cute::_128, cute::_64, cute::_8>,
        cute::Shape<cute::_2, cute::_1, cute::_1>,
        cute::Int<PIPELINE_STAGES>,
        cutlass::gemm::collective::KernelScheduleAuto
    >;
    using MMA = typename MMBuilder::TiledMma;
    using CollectiveMainloop = typename MMBuilder::CollectiveOp;
};
#endif //GEMM_CUH
