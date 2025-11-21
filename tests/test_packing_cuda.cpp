#include <array>
#include <vector>
#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "common/DualTrits.hpp"
#include "cpu/dual_trits_pack.hpp"
#include "cuda/dual_trits_pack.cuh"

namespace {

void SkipIfNoCudaDevice() {
    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "No CUDA device available: " << cudaGetErrorString(status);
    }
}

template <std::size_t Count>
using TritBlock = std::array<DualTrits, Count>;

template <std::size_t Count>
std::vector<DualTrits> Flatten(const std::vector<TritBlock<Count>>& blocks) {
    std::vector<DualTrits> flat;
    flat.reserve(blocks.size() * Count);
    for (auto const& block : blocks) {
        flat.insert(flat.end(), block.begin(), block.end());
    }
    return flat;
}

DualTrits MakeDualTrit(int exp, int dir) {
    return DualTrits(exp % DualTrits::BASE, static_cast<DualTrits::wide_t>(dir % DualTrits::BASE));
}

template <std::size_t Count>
TritBlock<Count> BlockFromSeed(std::uint64_t seed) {
    TritBlock<Count> block{};
    std::uint64_t idx = seed;
    for (std::size_t i = 0; i < Count; ++i) {
        auto dir = static_cast<int>(idx % DualTrits::BASE);
        idx /= DualTrits::BASE;
        auto exp = static_cast<int>(idx % DualTrits::BASE);
        idx /= DualTrits::BASE;
        block[i] = DualTrits(exp, static_cast<DualTrits::wide_t>(dir));
    }
    return block;
}

template <std::size_t Count, class UInt>
void ExpectRoundTrip(const std::vector<TritBlock<Count>>& blocks,
                     UInt (*cpu_pack)(DualTrits const*),
                     void (*cpu_unpack)(UInt, DualTrits*)) {
    const auto num_blocks = static_cast<int>(blocks.size());
    auto flat = Flatten(blocks);

    std::vector<UInt> packed(num_blocks);
    pack_dual_trits_batch_cuda<Count, UInt>(flat.data(), packed.data(), num_blocks);

    for (std::size_t i = 0; i < blocks.size(); ++i) {
        EXPECT_EQ(packed[i], cpu_pack(blocks[i].data())) << "Mismatch in pack result, block " << i;
    }

    std::vector<DualTrits> unpacked(blocks.size() * Count);
    unpack_dual_trits_batch_cuda<Count, UInt>(packed.data(), unpacked.data(), num_blocks);

    for (std::size_t i = 0; i < blocks.size(); ++i) {
        auto* out = unpacked.data() + (i * Count);
        for (std::size_t j = 0; j < Count; ++j) {
            EXPECT_EQ(blocks[i][j].getDirection(), out[j].getDirection()) << "block=" << i << " element=" << j;
            EXPECT_EQ(blocks[i][j].getExponent(),  out[j].getExponent())  << "block=" << i << " element=" << j;
        }
    }
}

} // namespace

TEST(CudaPack5, RoundTripMatchesCpu) {
    SkipIfNoCudaDevice();
    std::vector<TritBlock<5>> blocks = {
        { DualTrits(0,0), DualTrits(1,2), DualTrits(2,1), DualTrits(1,1), DualTrits(2,0) },
        { DualTrits(1,1), DualTrits(1,2), DualTrits(0,1), DualTrits(2,2), DualTrits(0,0) },
        BlockFromSeed<5>(42)
    };
    ExpectRoundTrip<5, std::uint16_t>(blocks, pack5, unpack5);
}

TEST(CudaPack10, RoundTripMatchesCpu) {
    SkipIfNoCudaDevice();
    std::vector<TritBlock<10>> blocks = {
        BlockFromSeed<10>(0),
        BlockFromSeed<10>(1),
        BlockFromSeed<10>(1234),
        BlockFromSeed<10>(9999)
    };
    ExpectRoundTrip<10, std::uint32_t>(blocks, pack10, unpack10);
}

TEST(CudaPack20, RoundTripMatchesCpu) {
    SkipIfNoCudaDevice();
    std::vector<TritBlock<20>> blocks;
    for (int i = 0; i < 3; ++i) {
        TritBlock<20> block{};
        for (int j = 0; j < 20; ++j) {
            block[j] = MakeDualTrit((i + j) % 3, (i + 2*j) % 3);
        }
        blocks.push_back(block);
    }
    blocks.push_back(BlockFromSeed<20>(555555));
    ExpectRoundTrip<20, std::uint64_t>(blocks, pack20, unpack20);
}
