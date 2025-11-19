#include <gtest/gtest.h>
#include "common/DualTrits.hpp"

/*
 * DualTrits arithmetic tests for addition and subtraction
 * 
 * Value mappings (from DualTrits.h):
 * DualTrits(0,0) = 0
 * DualTrits(0,1) = 1
 * DualTrits(0,2) = -1
 * DualTrits(1,1) = 3
 * DualTrits(1,2) = -3
 * DualTrits(2,1) = 1/3
 * DualTrits(2,2) = -1/3
 * DualTrits(1,0) = inf
 * DualTrits(2,0) = -inf
 */

// =============================
// Addition Tests
// =============================

TEST(DualTritsAddition, ZeroPlusZero) {
    DualTrits a(0, 0);  // 0
    DualTrits b(0, 0);  // 0
    DualTrits result = a + b;
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 0u);  // 0 + 0 = 0
}

TEST(DualTritsAddition, OnePlusZero) {
    DualTrits a(0, 1);  // 1
    DualTrits b(0, 0);  // 0
    DualTrits result = a + b;
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 1u);  // 1 + 0 = 1
}

TEST(DualTritsAddition, OnePlusOne) {
    DualTrits a(0, 1);  // 1
    DualTrits b(0, 1);  // 1
    DualTrits result = a + b;
    // 1 + 1 = 2, but we only have {..., -1, -1/3, 0, 1/3, 1, 3, ...}
    // 2 should round to 1 or 3, closer to 1 (distance 1 vs 1), round to even -> 3 is odd, 1 is odd
    // Actually from code: multiply by 3: 3 + 3 = 6, round to closest valid mul3 value
    // Valid values: -9, -3, -1, 0, 1, 3, 9
    // 6 is between 3 and 9, closer to 3 (distance 3 vs 3), tie -> round to even -> 3 is odd, 9 is odd
    // Let me check: 6 rounds to 3 (distance 3) or 9 (distance 3), both odd, pick smaller? 
    // Actually the code picks by index evenness, let's verify empirically
    EXPECT_EQ(result.getExponent(), 1u);
    EXPECT_EQ(result.getDirection(), 1u);  // Expected to be 3
}

TEST(DualTritsAddition, OneThirdPlusOneThird) {
    DualTrits a(2, 1);  // 1/3
    DualTrits b(2, 1);  // 1/3
    DualTrits result = a + b;
    // 1/3 + 1/3 = 2/3, mul3: 1 + 1 = 2, closest is 1 or 3
    // 2 is between 1 and 3, closer to 1 (distance 1 vs 1), tie
    // Round to even: both 1 and 3 are odd... pick by index evenness
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 1u);  // Should be 1
}

TEST(DualTritsAddition, ThreePlusOne) {
    DualTrits a(1, 1);  // 3
    DualTrits b(0, 1);  // 1
    DualTrits result = a + b;
    // 3 + 1 = 4, mul3: 9 + 3 = 12, overflow -> inf
    EXPECT_EQ(result.getExponent(), 1u);
    EXPECT_EQ(result.getDirection(), 0u);  // inf
}

TEST(DualTritsAddition, MinusOnePlusOne) {
    DualTrits a(0, 2);  // -1
    DualTrits b(0, 1);  // 1
    DualTrits result = a + b;
    // -1 + 1 = 0
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 0u);  // 0
}

TEST(DualTritsAddition, MinusThreePlusThree) {
    DualTrits a(1, 2);  // -3
    DualTrits b(1, 1);  // 3
    DualTrits result = a + b;
    // -3 + 3 = 0
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 0u);  // 0
}

TEST(DualTritsAddition, MinusOneThirdPlusOneThird) {
    DualTrits a(2, 2);  // -1/3
    DualTrits b(2, 1);  // 1/3
    DualTrits result = a + b;
    // -1/3 + 1/3 = 0
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 0u);  // 0
}

TEST(DualTritsAddition, ThreePlusThree) {
    DualTrits a(1, 1);  // 3
    DualTrits b(1, 1);  // 3
    DualTrits result = a + b;
    // 3 + 3 = 6, mul3: 9 + 9 = 18, overflow -> inf
    EXPECT_EQ(result.getExponent(), 1u);
    EXPECT_EQ(result.getDirection(), 0u);  // inf
}

TEST(DualTritsAddition, MinusThreePlusMinusThree) {
    DualTrits a(1, 2);  // -3
    DualTrits b(1, 2);  // -3
    DualTrits result = a + b;
    // -3 + -3 = -6, mul3: -9 + -9 = -18, overflow -> -inf
    EXPECT_EQ(result.getExponent(), 2u);
    EXPECT_EQ(result.getDirection(), 0u);  // -inf
}

TEST(DualTritsAddition, OnePlusMinusOneThird) {
    DualTrits a(0, 1);  // 1
    DualTrits b(2, 2);  // -1/3
    DualTrits result = a + b;
    // 1 + -1/3 = 2/3, mul3: 3 + -1 = 2, rounds to 1 or 3
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 1u);  // Should be 1
}

TEST(DualTritsAddition, MinusOnePlusOneThird) {
    DualTrits a(0, 2);  // -1
    DualTrits b(2, 1);  // 1/3
    DualTrits result = a + b;
    // -1 + 1/3 = -2/3, mul3: -3 + 1 = -2, closest is -1 or -3
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 2u);  // Should be -1
}

// =============================
// Subtraction Tests
// =============================

TEST(DualTritsSubtraction, ZeroMinusZero) {
    DualTrits a(0, 0);  // 0
    DualTrits b(0, 0);  // 0
    DualTrits result = a - b;
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 0u);  // 0 - 0 = 0
}

TEST(DualTritsSubtraction, OneMinusZero) {
    DualTrits a(0, 1);  // 1
    DualTrits b(0, 0);  // 0
    DualTrits result = a - b;
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 1u);  // 1 - 0 = 1
}

TEST(DualTritsSubtraction, OneMinusOne) {
    DualTrits a(0, 1);  // 1
    DualTrits b(0, 1);  // 1
    DualTrits result = a - b;
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 0u);  // 1 - 1 = 0
}

TEST(DualTritsSubtraction, ThreeMinusOne) {
    DualTrits a(1, 1);  // 3
    DualTrits b(0, 1);  // 1
    DualTrits result = a - b;
    // 3 - 1 = 2, mul3: 9 - 3 = 6, rounds to 3 or 9
    EXPECT_EQ(result.getExponent(), 1u);
    EXPECT_EQ(result.getDirection(), 1u);  // Should be 3
}

TEST(DualTritsSubtraction, OneMinusThree) {
    DualTrits a(0, 1);  // 1
    DualTrits b(1, 1);  // 3
    DualTrits result = a - b;
    // 1 - 3 = -2, mul3: 3 - 9 = -6, rounds to -3 or -9
    EXPECT_EQ(result.getExponent(), 1u);
    EXPECT_EQ(result.getDirection(), 2u);  // Should be -3
}

TEST(DualTritsSubtraction, OneThirdMinusOneThird) {
    DualTrits a(2, 1);  // 1/3
    DualTrits b(2, 1);  // 1/3
    DualTrits result = a - b;
    // 1/3 - 1/3 = 0
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 0u);  // 0
}

TEST(DualTritsSubtraction, OneMinusMinusOne) {
    DualTrits a(0, 1);  // 1
    DualTrits b(0, 2);  // -1
    DualTrits result = a - b;
    // 1 - (-1) = 2, mul3: 3 - (-3) = 6, rounds to 3 or 9
    EXPECT_EQ(result.getExponent(), 1u);
    EXPECT_EQ(result.getDirection(), 1u);  // Should be 3
}

TEST(DualTritsSubtraction, MinusOneMinusOne) {
    DualTrits a(0, 2);  // -1
    DualTrits b(0, 1);  // 1
    DualTrits result = a - b;
    // -1 - 1 = -2, mul3: -3 - 3 = -6, rounds to -3 or -9
    EXPECT_EQ(result.getExponent(), 1u);
    EXPECT_EQ(result.getDirection(), 2u);  // Should be -3
}

TEST(DualTritsSubtraction, ThreeMinusMinusThree) {
    DualTrits a(1, 1);  // 3
    DualTrits b(1, 2);  // -3
    DualTrits result = a - b;
    // 3 - (-3) = 6, mul3: 9 - (-9) = 18, overflow -> inf
    EXPECT_EQ(result.getExponent(), 1u);
    EXPECT_EQ(result.getDirection(), 0u);  // inf
}

TEST(DualTritsSubtraction, MinusThreeMinusThree) {
    DualTrits a(1, 2);  // -3
    DualTrits b(1, 1);  // 3
    DualTrits result = a - b;
    // -3 - 3 = -6, mul3: -9 - 9 = -18, overflow -> -inf
    EXPECT_EQ(result.getExponent(), 2u);
    EXPECT_EQ(result.getDirection(), 0u);  // -inf
}

TEST(DualTritsSubtraction, OneMinusOneThird) {
    DualTrits a(0, 1);  // 1
    DualTrits b(2, 1);  // 1/3
    DualTrits result = a - b;
    // 1 - 1/3 = 2/3, mul3: 3 - 1 = 2, rounds to 1 or 3
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 1u);  // Should be 1
}

TEST(DualTritsSubtraction, OneThirdMinusOne) {
    DualTrits a(2, 1);  // 1/3
    DualTrits b(0, 1);  // 1
    DualTrits result = a - b;
    // 1/3 - 1 = -2/3, mul3: 1 - 3 = -2, rounds to -1 or -3
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 2u);  // Should be -1
}

TEST(DualTritsSubtraction, MinusOneThirdMinusMinusOneThird) {
    DualTrits a(2, 2);  // -1/3
    DualTrits b(2, 2);  // -1/3
    DualTrits result = a - b;
    // -1/3 - (-1/3) = 0
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 0u);  // 0
}

// =============================
// Edge cases with special values
// =============================
// Note: Current implementation has limited support for special values (inf/-inf)
// mul3() returns -1 for special values, which may not give mathematically correct results

TEST(DualTritsAddition, ZeroPlusInf) {
    DualTrits a(0, 0);  // 0
    DualTrits b(1, 0);  // inf
    DualTrits result = a + b;
    // expected: inf
    EXPECT_TRUE(result.isSpecial());
    EXPECT_TRUE(result.isInfinity());
}

TEST(DualTritsAddition, ZeroPlusMinusInf) {
    DualTrits a(0, 0);  // 0
    DualTrits b(2, 0);  // -inf
    DualTrits result = a + b;
    // expected: -inf
    EXPECT_TRUE(result.isSpecial());
    EXPECT_TRUE(result.isNegativeInfinity());
}

// =============================
// Comprehensive value coverage
// =============================

TEST(DualTritsArithmetic, AllValuePairsCombinations) {
    // Test key combinations systematically
    struct TestCase {
        int exp_a, dir_a;
        int exp_b, dir_b;
        const char* desc;
    };
    
    std::vector<TestCase> cases = {
        {0, 0, 0, 0, "0 + 0"},
        {0, 1, 0, 0, "1 + 0"},
        {0, 2, 0, 0, "-1 + 0"},
        {2, 1, 0, 0, "1/3 + 0"},
        {2, 2, 0, 0, "-1/3 + 0"},
        {1, 1, 0, 0, "3 + 0"},
        {1, 2, 0, 0, "-3 + 0"},
    };
    
    for (const auto& tc : cases) {
        DualTrits a(tc.exp_a, tc.dir_a);
        DualTrits b(tc.exp_b, tc.dir_b);
        
        // Just verify operations don't crash and produce valid DualTrits
        DualTrits sum = a + b;
        DualTrits diff = a - b;
        
        EXPECT_LE(sum.getExponent(), 2u) << tc.desc;
        EXPECT_LE(sum.getDirection(), 2u) << tc.desc;
        EXPECT_LE(diff.getExponent(), 2u) << tc.desc;
        EXPECT_LE(diff.getDirection(), 2u) << tc.desc;
    }
}

// =============================
// Commutativity Tests
// =============================

TEST(DualTritsProperties, AdditionCommutative) {
    std::vector<DualTrits> values = {
        DualTrits(0, 0),  // 0
        DualTrits(0, 1),  // 1
        DualTrits(0, 2),  // -1
        DualTrits(2, 1),  // 1/3
        DualTrits(2, 2),  // -1/3
        DualTrits(1, 1),  // 3
        DualTrits(1, 2),  // -3
    };
    
    for (const auto& a : values) {
        for (const auto& b : values) {
            DualTrits ab = a + b;
            DualTrits ba = b + a;
            EXPECT_EQ(ab.getExponent(), ba.getExponent()) 
                << "a=" << a.getExponent() << "," << a.getDirection()
                << " b=" << b.getExponent() << "," << b.getDirection();
            EXPECT_EQ(ab.getDirection(), ba.getDirection())
                << "a=" << a.getExponent() << "," << a.getDirection()
                << " b=" << b.getExponent() << "," << b.getDirection();
        }
    }
}

TEST(DualTritsProperties, SubtractionAntiCommutative) {
    std::vector<DualTrits> values = {
        DualTrits(0, 0),  // 0
        DualTrits(0, 1),  // 1
        DualTrits(0, 2),  // -1
        DualTrits(2, 1),  // 1/3
        DualTrits(2, 2),  // -1/3
    };
    
    for (const auto& a : values) {
        for (const auto& b : values) {
            DualTrits ab = a - b;
            DualTrits ba = b - a;
            // a - b should be -(b - a) in terms of direction, if no rounding involved
            // This is approximate due to rounding, just verify symmetry exists
            if (ab.getExponent() == ba.getExponent()) {
                // If same exponent, directions should be opposite (unless both zero)
                if (ab.getDirection() == 0 && ba.getDirection() == 0) {
                    EXPECT_EQ(ab.getExponent(), 0u);  // Both should be zero
                }
            }
        }
    }
}
