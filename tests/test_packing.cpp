#include <gtest/gtest.h>
#include "common/DualTrits.hpp"
#include "cpu/dual_trits_pack.hpp"

// Helper: decode an integer into 10 base-3 digits and fill 5 DualTrits
static void fill_dual_trits_from_index(uint32_t idx, DualTrits (&arr)[5]) {
    // order per element: direction (least significant), then exponent
    for (int i = 0; i < 5; ++i) {
        auto dir = static_cast<int>(idx % DualTrits::BASE);
        idx /= DualTrits::BASE;
        auto exp = static_cast<int>(idx % DualTrits::BASE);
        idx /= DualTrits::BASE;
        arr[i] = DualTrits(exp, static_cast<DualTrits::wide_t>(dir));
    }
}

TEST(Pack5, RoundTrip_Simple) {
    DualTrits in[5] = {
        DualTrits(0, 0),
        DualTrits(1, 2),
        DualTrits(2, 1),
        DualTrits(1, 1),
        DualTrits(2, 0)
    };

    std::uint16_t packed = pack5(in);
    DualTrits out[5] { DualTrits(), DualTrits(), DualTrits(), DualTrits(), DualTrits() };
    unpack5(packed, out);

    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(in[i].getDirection(), out[i].getDirection()) << "mismatch at i=" << i << " (direction)";
        EXPECT_EQ(in[i].getExponent() , out[i].getExponent() ) << "mismatch at i=" << i << " (exponent)";
    }
}

TEST(Pack5, RoundTrip_AllZeros) {
    DualTrits in[5] = { DualTrits(0,0), DualTrits(0,0), DualTrits(0,0), DualTrits(0,0), DualTrits(0,0) };
    auto packed = pack5(in);
    DualTrits out[5] { DualTrits(), DualTrits(), DualTrits(), DualTrits(), DualTrits() };
    unpack5(packed, out);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(out[i].getDirection(), 0u);
        EXPECT_EQ(out[i].getExponent() , 0u);
    }
}

TEST(Pack5, RoundTrip_Exhaustive_59049) {
    // 5 elements * 2 trits each => 10 trits; 3^10 combinations = 59049
    const uint32_t combinations = 59049; // 3^10
    for (uint32_t idx = 0; idx < combinations; ++idx) {
        DualTrits in[5] { DualTrits(), DualTrits(), DualTrits(), DualTrits(), DualTrits() };
        fill_dual_trits_from_index(idx, in);

        auto packed = pack5(in);
        DualTrits out[5] { DualTrits(), DualTrits(), DualTrits(), DualTrits(), DualTrits() };
        unpack5(packed, out);

        for (int i = 0; i < 5; ++i) {
            ASSERT_EQ(in[i].getDirection(), out[i].getDirection()) << "idx=" << idx << ", i=" << i;
            ASSERT_EQ(in[i].getExponent() , out[i].getExponent() ) << "idx=" << idx << ", i=" << i;
        }
    }
}

// =============================
// pack10 / unpack10 test cases
// =============================
TEST(Pack10, RoundTrip_Simple) {
    DualTrits in[10];
    for (int i = 0; i < 10; ++i) {
        in[i] = DualTrits(i % 3, static_cast<DualTrits::wide_t>((i+1) % 3));
    }
    auto packed = pack10(in);
    DualTrits out[10]{};
    unpack10(packed, out);
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(in[i].getDirection(), out[i].getDirection()) << "i=" << i;
        EXPECT_EQ(in[i].getExponent(),  out[i].getExponent())  << "i=" << i;
    }
}

TEST(Pack10, RoundTrip_AllZeros) {
    DualTrits in[10];
    for (auto & d : in) d = DualTrits(0,0);
    auto packed = pack10(in);
    DualTrits out[10]{};
    unpack10(packed, out);
    for (auto & d : out) {
        EXPECT_EQ(d.getExponent(), 0u);
        EXPECT_EQ(d.getDirection(), 0u);
    }
}

// For pack10 exhaustive would be 3^(20) ≈ 3.486e9 combinations (too large), so we sample.
TEST(Pack10, RoundTrip_SampledPatterns) {
    DualTrits in[10];
    // 6 chosen pattern seeds
    int seeds[] = {0, 1, 5, 40, 242, 728};
    for (int s : seeds) {
        int idx = s;
        for (int i = 0; i < 10; ++i) {
            auto dir = static_cast<int>(idx % DualTrits::BASE);
            idx /= DualTrits::BASE;
            auto exp = static_cast<int>(idx % DualTrits::BASE);
            idx /= DualTrits::BASE;
            in[i] = DualTrits(exp, static_cast<DualTrits::wide_t>(dir));
        }
        auto packed = pack10(in);
        DualTrits out[10]{};
        unpack10(packed, out);
        for (int i = 0; i < 10; ++i) {
            ASSERT_EQ(in[i].getDirection(), out[i].getDirection()) << "seed=" << s << ", i=" << i;
            ASSERT_EQ(in[i].getExponent(),  out[i].getExponent())  << "seed=" << s << ", i=" << i;
        }
    }
}

// =============================
// pack20 / unpack20 test cases
// =============================
TEST(Pack20, RoundTrip_Simple) {
    DualTrits in[20];
    for (int i = 0; i < 20; ++i) {
        in[i] = DualTrits((i*2) % 3, static_cast<DualTrits::wide_t>((i+2) % 3));
    }
    auto packed = pack20(in);
    DualTrits out[20]{};
    unpack20(packed, out);
    for (int i = 0; i < 20; ++i) {
        EXPECT_EQ(in[i].getDirection(), out[i].getDirection()) << "i=" << i;
        EXPECT_EQ(in[i].getExponent(),  out[i].getExponent())  << "i=" << i;
    }
}

TEST(Pack20, RoundTrip_AllZeros) {
    DualTrits in[20];
    for (auto & d : in) d = DualTrits(0,0);
    auto packed = pack20(in);
    DualTrits out[20]{};
    unpack20(packed, out);
    for (auto & d : out) {
        EXPECT_EQ(d.getExponent(), 0u);
        EXPECT_EQ(d.getDirection(), 0u);
    }
}

// Sample a handful of pseudo-random patterns for pack20
TEST(Pack20, RoundTrip_SampledPatterns) {
    DualTrits in[20]{};
    // seeds chosen to exercise varied digit distributions
    unsigned long long seeds[] = {0ULL, 1ULL, 12345ULL, 555555ULL, 1048575ULL};
    for (auto s : seeds) {
        unsigned long long idx = s;
        for (int i = 0; i < 20; ++i) {
            auto dir = static_cast<int>(idx % DualTrits::BASE);
            idx /= DualTrits::BASE;
            auto exp = static_cast<int>(idx % DualTrits::BASE);
            idx /= DualTrits::BASE;
            in[i] = DualTrits(exp, static_cast<DualTrits::wide_t>(dir));
        }
        auto packed = pack20(in);
        DualTrits out[20]{};
        unpack20(packed, out);
        for (int i = 0; i < 20; ++i) {
            ASSERT_EQ(in[i].getDirection(), out[i].getDirection()) << "seed=" << s << ", i=" << i;
            ASSERT_EQ(in[i].getExponent(),  out[i].getExponent())  << "seed=" << s << ", i=" << i;
        }
    }
}
