//
// Created by ubuntu on 2/9/25.
//

#ifndef HGEMM_CUH
#define HGEMM_CUH

#include <cute/tensor.hpp>
#include <cute/layout.hpp>

#include "gemm.cuh"
#include "util.cuh"

namespace abacus {
    enum class ResultType {
        local,
        network
    };

    #define MULTIPLE_TIMING 0
    template<class BlockGEMM, unsigned int sharedSize = 16 * 1024, ResultType r = ResultType::local,
    typename ProblemShape, typename StrideAB>
    requires (sharedSize % THREADS == 0 && rank(ProblemShape{}) == 4)
    __global__ __maxnreg__(128) __cluster_dims__(2, 1, 1)
    void deviceCollectiveMMA(ProblemShape pS, StrideAB sAB,
        const typename BlockGEMM::MatrixAType* __restrict__ pA,
        const typename BlockGEMM::MatrixBType* __restrict__ pB, typename BlockGEMM::MatrixDType* __restrict__ pC,
        const typename BlockGEMM::MatrixDType* __restrict__ pD, const bool skip = true) {
#if MULTIPLE_TIMING
        float clocked = 0.0f;
        constexpr auto rounds = 8;
        for (uint k = 0; k < rounds; ++k) {
            uint64_t start = 0, end = 0;
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
#endif
            auto [M, N, K, L] = pS;
            using CollectiveMMAType = typename BlockGEMM::CollectiveMainloop;
            auto mA = cute::make_tensor(CAST_TO(typename CollectiveMMAType::InternalElementA, pA),
                cute::make_layout(cute::make_shape(M, K, L), sAB));
            auto mB = cute::make_tensor(CAST_TO(typename CollectiveMMAType::InternalElementA, pB),
                cute::make_layout(cute::make_shape(N, K, L), sAB));

            typename CollectiveMMAType::Params::TMA_A tma_load_a = cute::make_tma_copy(
                typename CollectiveMMAType::GmemTiledCopyA{},
                mA,
                typename CollectiveMMAType::SmemLayoutA{}(cute::_,cute::_,cute::Int<0>{}),
                cute::make_shape(cute::shape<0>(typename CollectiveMMAType::TileShape{}),
                    cute::shape<2>(typename CollectiveMMAType::TileShape{})),
                    cute::size<1>(typename CollectiveMMAType::ClusterShape{}));

            typename CollectiveMMAType::Params::TMA_A tma_load_b = cute::make_tma_copy(
                typename CollectiveMMAType::GmemTiledCopyB{},
                mB,
                typename CollectiveMMAType::SmemLayoutB{}(cute::_,cute::_,cute::Int<0>{}),
                cute::make_shape(cute::shape<1>(typename CollectiveMMAType::TileShape{}),
                    cute::shape<2>(typename CollectiveMMAType::TileShape{})),
                    cute::size<0>(typename CollectiveMMAType::ClusterShape{}));

            auto cP = typename CollectiveMMAType::Params{
                tma_load_a, tma_load_b
            };

            auto mA_mkl = tma_load_a.get_tma_tensor(make_shape(M,K,L));
            auto mB_nkl = tma_load_b.get_tma_tensor(make_shape(N,K,L));

            const auto mC = make_tensor(cute::make_gmem_ptr(pC),
                make_layout(cute::make_shape(M, N), cute::make_stride(N, 1)));
            // bias vector (1, N) broadcast to (M, N)
            const auto mD = make_tensor(cute::make_gmem_ptr(pD),
                make_layout(cute::make_shape(M, N), cute::make_stride(0, 1)));

            using ElementD = typename BlockGEMM::MatrixDType;
            static_assert(sharedSize % sizeof(ElementD) == 0);
            __shared__ __align__(16) ElementD scratch[sharedSize / sizeof(ElementD)];

            using ElementC = typename BlockGEMM::MatrixCType;
            constexpr auto bM = cute::get<0>(typename BlockGEMM::BlockTiler{});
            constexpr auto bN = cute::get<0>(typename BlockGEMM::BlockTiler{});
            constexpr typename BlockGEMM::MMA tiledMMA{};
            auto accum = cute::partition_fragment_C(tiledMMA, typename BlockGEMM::TilerOut{});
            static_assert(cuda::std::is_same_v<ElementC, typename decltype(accum)::value_type>);
            // Get the appropriate tiles for this thread block
            const auto cta_coordX = cute::idx2crd(blockIdx.x, cute::Shape(cute::ceil_div(cute::get<0>(mC.shape()), bM),
                cute::ceil_div(cute::get<1>(mC.shape()), bN)));

            const auto cta_coord = cute::make_coord(cute::get<0>(cta_coordX), cute::get<1>(cta_coordX), cute::_);
            const auto gA = local_tile(mA, typename BlockGEMM::BlockTiler{}, cta_coord, cute::Step<cute::_1, cute::X,cute::_1>{});  // (BLK_M,BLK_K,k)
            const auto gB = local_tile(mB, typename BlockGEMM::BlockTiler{}, cta_coord, cute::Step< cute::X,cute::_1,cute::_1>{});  // (BLK_N,BLK_K,k)
            const auto gC = local_tile(mC, typename BlockGEMM::BlockTiler{}, cta_coord, cute::Step<cute::_1,cute::_1, cute::X>{});  // (BLK_M,BLK_N)
            const auto gD = local_tile(mD, typename BlockGEMM::BlockTiler{}, cta_coord, cute::Step<cute::_1,cute::_1, cute::X>{});  // (BLK_M,BLK_N)
            auto k_tile_iter = cute::make_coord_iterator(size<2>(gA));
            int k_tile_count = size<2>(gA);
            cute::clear(accum);
            typename BlockGEMM::CollectiveMainloop mainLoop{};
            constexpr auto cMp = typename BlockGEMM::CollectiveMainloop::Params{};
            mainLoop(
                gA,
                cMp.tma_load_a,
                gB,
                cMp.tma_load_b,
                accum,
                k_tile_iter, k_tile_count,
                threadIdx.x,
                cute::block_rank_in_cluster(),
                CAST_TO(char, scratch),
                cMp);

            // Epilogue
            //const auto tCgC = tiledMMA.get_slice(threadIdx.x).partition_C(gC);
            const auto tDgD = tiledMMA.get_slice(threadIdx.x).partition_C(gD);
            constexpr auto gCStoreOp = cutlass::NumericConverter<typename decltype(gC)::value_type, ElementC>{};
            constexpr auto gDLoadOp = cutlass::NumericConverter<ElementC, ElementD>{};

            // Assume unary operator
            constexpr typename BlockGEMM::FusedEpilogue epilogueOp{};
            constexpr auto elems = sharedSize / (THREADS * sizeof(ElementD));
            static_assert(size(accum) % elems == 0);
            constexpr auto trips = size(accum) / elems;
            // Leverage compiler packing for half-precision values into one register
            cutlass::AlignedArray<ElementD, elems> rScratch{};

            // Prefetch from global to shared memory
            #pragma unroll
            for (int j = 0; j < elems; ++j) {
                scratch[threadIdx.x + j * THREADS] = tDgD(j);
            }

            #pragma unroll
            for (unsigned int i = 0; i < trips; ++i) {
                #pragma unroll
                for (unsigned int j = 0; j < elems; ++j) {
                    rScratch[j] = scratch[threadIdx.x + j * THREADS];
                    if (i + 1 < trips) {
                        // Eagerly start loads for the next batch, if needed
                        scratch[threadIdx.x + j * THREADS] = tDgD(j + (i + 1) * elems);
                    }
                }
                // Fused Bias Add and Activation Function on register fragment
                // Also fuses copy to GMEM.
                #pragma unroll
                for (int j = 0; j < elems; ++j) {
                    accum(j + i * elems) = epilogueOp(accum(j + i * elems), gDLoadOp(rScratch[j]));
                }
            }

            __syncthreads();
            constexpr auto sCLay = cute::make_layout(cute::Shape<cute::Int<bM>, cute::Int<elems>>{}, cute::LayoutRight{});
            const auto sC = cute::make_tensor(cute::make_smem_ptr(scratch), sCLay);
            const auto tCsC = tiledMMA.get_slice(threadIdx.x).partition_C(sC);
            const auto rIdx = threadIdx.x / elems * elems;
            const auto cIdx = threadIdx.x % elems;
            #pragma unroll
            for (unsigned int i = 0; i < trips; ++i) {
                #pragma unroll
                for (unsigned int j = 0; j < elems; ++j) {
                    tCsC(j) = gCStoreOp(accum(j + i * elems));
                }
                __syncthreads();
                #pragma unroll
                for (unsigned int j = 0; j < elems; ++j) {
                    gC(rIdx + j, cIdx + i * elems) = sC(rIdx + j, cIdx);
                }
            }
            __syncthreads();
            if (!threadIdx.x) {
                if constexpr (r == ResultType::local) {
                    __threadfence();
                }
                else {
                    __threadfence_system();
                }
            }
            __syncthreads();
#if MULTIPLE_TIMING
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
            clocked += static_cast<float>(end - start) / static_cast<float>(rounds);
        }
        if (!threadIdx.x && !skip) {
            printf("Duration was %fus\n", clocked / 1000.0f);
        }
#endif

#if 0
        if (!threadIdx.x && !skip) {
            cute::print_tensor(mD);
            cute::print_tensor(mA);
            cute::print_tensor(mB);
            cute::print_tensor(mC);
        }
#endif
    }

    __host__ __forceinline__
    void testCollective() {
        const auto playStream = cudaStreamPerThread;
        constexpr auto M = 128;
        constexpr auto N = 64;
        constexpr auto K = 64;

        using inputValueType = cute::tfloat32_t;
        using outValueType = inputValueType;
        using weightValueType = inputValueType;
        using accumulateType = float;

        using activation = cutlass::epilogue::thread::ReLU<accumulateType>;
        using Operation = BlockMM<900, inputValueType, weightValueType, accumulateType, activation>;
        constexpr auto aSize = (sizeof(inputValueType) * M * K);
        constexpr auto abSize = aSize + (sizeof(weightValueType) * N * K);
        constexpr auto abcSize = abSize + (sizeof(outValueType) * M * N);
        constexpr auto len = abcSize + (sizeof(inputValueType) * cute::max(N, K));
        cuda::std::byte* abc;
        CHECK_ERROR_EXIT(cudaMallocAsync(&abc, len, playStream));
        CHECK_ERROR_EXIT(cudaMemsetAsync(abc, 0, len, playStream));
        auto* data = static_cast<cuda::std::byte*>(calloc(len, sizeof(cuda::std::byte)));

        auto mAHost = make_tensor(CAST_TO(inputValueType, data),
            make_layout(cute::make_shape(M, K), cute::make_stride(K, 1)));
        auto mBHost = make_tensor(CAST_TO(weightValueType, data + aSize),
            make_layout(cute::make_shape(N, K), cute::make_stride(K, 1)));

        // Populate bias vector
        CAST_TO(inputValueType, data + abcSize)[0] = static_cast<inputValueType>(1.0);
        CAST_TO(inputValueType, data + abcSize)[1] = static_cast<inputValueType>(2.0);

        mAHost(0, 0) = static_cast<inputValueType>(0.0);
        mAHost(0, 1) = static_cast<inputValueType>(1.0);
        mAHost(1, 0) = static_cast<inputValueType>(2.0);
        mAHost(1, 1) = static_cast<inputValueType>(3.0);

        mBHost(0, 0) = static_cast<weightValueType>(4.0);
        mBHost(0, 1) = static_cast<weightValueType>(5.0);
        mBHost(1, 0) = static_cast<weightValueType>(6.0);
        mBHost(1, 1) = static_cast<weightValueType>(7.0);

        CHECK_ERROR_EXIT(cudaMemcpyAsync(abc, data, len, cudaMemcpyHostToDevice, playStream));

        const auto mA = make_tensor(cute::make_gmem_ptr(CAST_TO(inputValueType, abc)),
            make_layout(cute::make_shape(M, K), cute::make_stride(K, 1)));
        const auto mB = make_tensor(cute::make_gmem_ptr(
            CAST_TO(weightValueType, abc + aSize)), make_layout(cute::make_shape(N, K), cute::make_stride(K, 1)));
        const auto mC = make_tensor(cute::make_gmem_ptr(CAST_TO(inputValueType, abc + abSize)),
            make_layout(cute::make_shape(M, N), cute::make_stride(N, 1)));

        // bias vector (1, N) broadcast to (M, N)
        const auto mD = make_tensor(cute::make_gmem_ptr(CAST_TO(inputValueType, abc + abcSize)),
            make_layout(cute::make_shape(M, N), cute::make_stride(0, 1)));

        constexpr auto gemmSharedSize = (sizeof(inputValueType) * Operation::GEMM::a_size)
            + (sizeof(weightValueType) + Operation::GEMM::b_size);
        constexpr auto sharedSize = cute::max(gemmSharedSize * PIPELINE_STAGES, 128 * 32 * 4);
        deviceCollectiveMMA<Operation, sharedSize><<<1, 128, 0, playStream>>>(mA, mB, mC, mD, false);
        CHECK_LAST();
        CHECK_ERROR_EXIT(cudaFree(abc));
        free(data);
    }
}
#endif //HGEMM_CUH
