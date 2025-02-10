//
// Created by ubuntu on 2/9/25.
//

#ifndef GEMM_CUH
#define GEMM_CUH

#include <cublasdx.hpp>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/gemm/collective/collective_mma.hpp>

#define MIN_ARCH 700U
#define THREADS 128U
#define BLOCK_M 128U
#define BLOCK_N 64U
#define BLOCK_K_HALF 16U
#define BLOCK_K_FULL 8U
#define MAX_REGS (BLOCK_M * BLOCK_N) / THREADS
#define PIPELINE_STAGES 2U

namespace abacus {
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

    /// Fused, Add, Activate
    template <typename Element, typename ActivationFunction>
    requires(TensorValueType<Element> && cuda::std::is_invocable_r_v<Element, ActivationFunction, Element>)
    struct FAA {
        // fp8
        __forceinline__ __device__
        Element operator()(const Element& accumulator, const Element& term) const {
            constexpr ActivationFunction op{};
            return op(accumulator + term);
        }
    };

    // specialization for half-precision and relu
    template<>
    struct FAA<cute::half_t, cutlass::epilogue::thread::ReLU<cute::half_t>> {
        __forceinline__ __device__
        cute::half_t operator()(const cute::half_t& accumulator, const cute::half_t& term) const {
            return cute::half_t(__hfma_relu(__half(1.0f),accumulator.to_half(), term.to_half()));
        }
    };

    // specialization for bfloat16 and relu
    template<>
    struct FAA<cute::bfloat16_t, cutlass::epilogue::thread::ReLU<cute::bfloat16_t>> {
        __forceinline__ __device__
        cute::bfloat16_t operator()(const cute::bfloat16_t& accumulator, const cute::bfloat16_t& term) const {
            return cute::bfloat16_t(__hfma_relu(__nv_bfloat16(1.0f),
                accumulator.to_nv_bfloat16(), term.to_nv_bfloat16()));
        }
    };

    template<typename F>
    struct isFAA : cuda::std::false_type {};

    template<typename Element, typename ActivationFunction>
    struct isFAA<FAA<Element, ActivationFunction>> : cuda::std::true_type {};

    // Source: CUTLASS
    // Maps a rank-1 cute::Shape<> representing the cluster shape on to the TMA atom that should be used with it
    template <class UnimodalClusterShape>
    constexpr auto
    sm90_cluster_shape_to_tma_atom(UnimodalClusterShape) {
        static_assert(rank(UnimodalClusterShape{}) == 1,
          "Use this function to figure out TMA for each mode individually.");

        if constexpr (cute::size(UnimodalClusterShape{}) == 1) {
            return cute::SM90_TMA_LOAD{};
        }
        else {
            return cute::SM90_TMA_LOAD_MULTICAST{};
        }
    }

    // Source: CUTLASS
    // Helper for SS GMMA smem selection that considers a tensor TileShape:
    //   (BLK_MN, BLK_K)
    //   or hierarchically
    //   ((BLK_MN0,BLK_MN1,...),(BLK_K0,BLK_K1,...))
    //   and returns the largest GMMA::Layout that fits BLK_MN0 and BLK_K0
    template <cute::GMMA::Major major, class ElementType, class BLK_MN, class BLK_K>
    CUTE_HOST_DEVICE constexpr
    auto
    ss_smem_selector()
    {
      using namespace cute;

      auto BLK_MN0 = size<0>(BLK_MN{});
      auto BLK_K0  = size<0>(BLK_K{});

      static_assert(BLK_MN0 % 8 == 0, "BLK_MN0 must be a multiple of 8.");
      static_assert(BLK_K0 % 8 == 0,  "BLK_K0 must be a multiple of 8.");

      if constexpr (major == GMMA::Major::MN) {
        if constexpr (BLK_MN0 % size<0>(GMMA::Layout_MN_SW128_Atom<ElementType>{}) == 0) {
          return GMMA::Layout_MN_SW128_Atom<ElementType>{};
        }
        else if constexpr (BLK_MN0 % size<0>(GMMA::Layout_MN_SW64_Atom<ElementType>{}) == 0) {
          return GMMA::Layout_MN_SW64_Atom<ElementType>{};
        }
        else if constexpr (BLK_MN0 % size<0>(GMMA::Layout_MN_SW32_Atom<ElementType>{}) == 0) {
          return GMMA::Layout_MN_SW32_Atom<ElementType>{};
        }
        else if constexpr (BLK_MN0 % size<0>(GMMA::Layout_MN_INTER_Atom<ElementType>{}) == 0) {
          return GMMA::Layout_MN_INTER_Atom<ElementType>{};
        }
        else {
          static_assert(BLK_MN0 % size<0>(GMMA::Layout_MN_INTER_Atom<ElementType>{}) == 0,
                        "BLK_MN0 must be a multiple of size<0>(GMMA::Layout_MN_INTER_Atom<ElementType>{})");
        }
      }
      else if constexpr (major == GMMA::Major::K) {
        if constexpr (BLK_K0 % size<1>(GMMA::Layout_K_SW128_Atom<ElementType>{}) == 0) {
          return GMMA::Layout_K_SW128_Atom<ElementType>{};
        }
        else if constexpr (BLK_K0 % size<1>(GMMA::Layout_K_SW64_Atom<ElementType>{}) == 0) {
          return GMMA::Layout_K_SW64_Atom<ElementType>{};
        }
        else if constexpr (BLK_K0 % size<1>(GMMA::Layout_K_SW32_Atom<ElementType>{}) == 0) {
          return GMMA::Layout_K_SW32_Atom<ElementType>{};
        }
        else if constexpr (BLK_K0 % size<1>(GMMA::Layout_K_INTER_Atom<ElementType>{}) == 0) {
          return GMMA::Layout_K_INTER_Atom<ElementType>{};
        }
        else {
          static_assert(BLK_K0 % size<1>(GMMA::Layout_K_INTER_Atom<ElementType>{}) == 0,
                        "BLK_K0 must be a multiple of size<1>(GMMA::Layout_K_INTER_Atom<ElementType>{})");
        }
      }
    }

    // Hopper only
    template<
        typename ElementA,
        typename ElementB,
        typename ElementC,
        typename ActivationOp
    >
    struct BlockMM {
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
        using TilerOut = cute::Shape<cute::Int<cublasdx::size_of<GEMM>::m>, cute::Int<cublasdx::size_of<GEMM>::n>>;

        // Construct CollectiveMMA
        using GmemLayoutATag = cutlass::layout::RowMajor;
        using GmemLayoutBTag = cutlass::layout::RowMajor;
        using ClusterShape_MNK = cute::Shape<cute::_2,cute::_1,cute::_1>;

        using ElementAMma = cute::conditional_t<cute::is_same_v<ElementA, float>, cute::tfloat32_t, ElementA>;
        using ElementBMma = cute::conditional_t<cute::is_same_v<ElementB, float>, cute::tfloat32_t, ElementB>;

        using MMA = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<
          ElementAMma, ElementBMma, ElementC, BlockTiler, cute::GMMA::Major::K, cute::GMMA::Major::K>()));

        using GmemTiledCopyA = decltype(sm90_cluster_shape_to_tma_atom(cute::shape<1>(ClusterShape_MNK{})));
        using GmemTiledCopyB = decltype(sm90_cluster_shape_to_tma_atom(cute::shape<0>(ClusterShape_MNK{})));

        using SmemLayoutAtomA = decltype(ss_smem_selector<
          cute::GMMA::Major::K, ElementAMma, decltype(cute::get<0>(BlockTiler{})), decltype(cute::get<2>(BlockTiler{}))>());
        using SmemLayoutAtomB = decltype(ss_smem_selector<
            cute::GMMA::Major::K, ElementBMma, decltype(cute::get<1>(BlockTiler{})), decltype(cute::get<2>(BlockTiler{}))>());

        using DispatchPolicy = cutlass::gemm::MainloopSm90TmaGmma<4, ClusterShape_MNK>;
        using SmemCopyAtomA = void;
        using SmemCopyAtomB = void;

        // Ensures compatibility with Cute
        using StrideIntType = int64_t;
        using RowMajorStride = cute::Stride<StrideIntType, cute::Int<1>, StrideIntType>;
        using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
          DispatchPolicy,
          BlockTiler,
          ElementA,
          RowMajorStride,
          ElementB,
          RowMajorStride,
          MMA,
          GmemTiledCopyA,
          SmemLayoutAtomA,
          SmemCopyAtomA,
          cute::identity,
          GmemTiledCopyB,
          SmemLayoutAtomB,
          SmemCopyAtomB,
          cute::identity
        >;
        using FusedEpilogue = FAA<ElementC, ActivationOp>;
    };
}
#endif //GEMM_CUH
