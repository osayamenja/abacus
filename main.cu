#include "hGemm.cuh"

#include <cute/tensor.hpp>
#include "gemm.cuh"

template<typename BlockGEMM>
__global__ void theatre() {
    __shared__ __align__(16) typename BlockGEMM::MatrixAType scratch[BlockGEMM::GEMM::a_size];
    auto sA = cute::make_tensor(cute::make_smem_ptr(scratch), typename BlockGEMM::CollectiveMainloop::SmemLayoutA{});
}

__host__ __forceinline__
void hostT() {
    using inputValueType = cute::tfloat32_t;
    using outValueType = inputValueType;
    using weightValueType = inputValueType;
    using accumulateType = float;

    using activation = cutlass::epilogue::thread::ReLU<accumulateType>;
    using Operation = abacus::BlockMM<inputValueType, weightValueType, accumulateType, activation>;
    theatre<Operation><<<1,1>>>();
}
int main() {
    abacus::testCollective();
    return 0;
}