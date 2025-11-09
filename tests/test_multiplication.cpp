#include <gtest/gtest.h>
#include "DualTrits.h"

/*
 * DualTrits multiplication tests
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
// Basic Multiplication Tests
// =============================

TEST(DualTritsMultiplication, ZeroTimesZero) {
    DualTrits a(0, 0);  // 0
    DualTrits b(0, 0);  // 0
    DualTrits result = a * b;
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 0u);  // 0 * 0 = 0
}

TEST(DualTritsMultiplication, OneTimesZero) {
    DualTrits a(0, 1);  // 1
    DualTrits b(0, 0);  // 0
    DualTrits result = a * b;
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 0u);  // 1 * 0 = 0
}

TEST(DualTritsMultiplication, ZeroTimesThree) {
    DualTrits a(0, 0);  // 0
    DualTrits b(1, 1);  // 3
    DualTrits result = a * b;
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 0u);  // 0 * 3 = 0
}

TEST(DualTritsMultiplication, OneTimesOne) {
    DualTrits a(0, 1);  // 1
    DualTrits b(0, 1);  // 1
    DualTrits result = a * b;
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 1u);  // 1 * 1 = 1
}

TEST(DualTritsMultiplication, OneTimesMinusOne) {
    DualTrits a(0, 1);  // 1
    DualTrits b(0, 2);  // -1
    DualTrits result = a * b;
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 2u);  // 1 * -1 = -1
}

TEST(DualTritsMultiplication, MinusOneTimesMinusOne) {
    DualTrits a(0, 2);  // -1
    DualTrits b(0, 2);  // -1
    DualTrits result = a * b;
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 1u);  // -1 * -1 = 1
}

TEST(DualTritsMultiplication, OneTimesThree) {
    DualTrits a(0, 1);  // 1
    DualTrits b(1, 1);  // 3
    DualTrits result = a * b;
    EXPECT_EQ(result.getExponent(), 1u);
    EXPECT_EQ(result.getDirection(), 1u);  // 1 * 3 = 3
}

TEST(DualTritsMultiplication, ThreeTimesOne) {
    DualTrits a(1, 1);  // 3
    DualTrits b(0, 1);  // 1
    DualTrits result = a * b;
    EXPECT_EQ(result.getExponent(), 1u);
    EXPECT_EQ(result.getDirection(), 1u);  // 3 * 1 = 3
}

TEST(DualTritsMultiplication, ThreeTimesMinusOne) {
    DualTrits a(1, 1);  // 3
    DualTrits b(0, 2);  // -1
    DualTrits result = a * b;
    EXPECT_EQ(result.getExponent(), 1u);
    EXPECT_EQ(result.getDirection(), 2u);  // 3 * -1 = -3
}

TEST(DualTritsMultiplication, MinusThreeTimesOne) {
    DualTrits a(1, 2);  // -3
    DualTrits b(0, 1);  // 1
    DualTrits result = a * b;
    EXPECT_EQ(result.getExponent(), 1u);
    EXPECT_EQ(result.getDirection(), 2u);  // -3 * 1 = -3
}

TEST(DualTritsMultiplication, MinusThreeTimesMinusOne) {
    DualTrits a(1, 2);  // -3
    DualTrits b(0, 2);  // -1
    DualTrits result = a * b;
    EXPECT_EQ(result.getExponent(), 1u);
    EXPECT_EQ(result.getDirection(), 1u);  // -3 * -1 = 3
}

// =============================
// Fraction Multiplication Tests
// =============================

TEST(DualTritsMultiplication, OneThirdTimesOne) {
    DualTrits a(2, 1);  // 1/3
    DualTrits b(0, 1);  // 1
    DualTrits result = a * b;
    EXPECT_EQ(result.getExponent(), 2u);
    EXPECT_EQ(result.getDirection(), 1u);  // 1/3 * 1 = 1/3
}

TEST(DualTritsMultiplication, OneThirdTimesThree) {
    DualTrits a(2, 1);  // 1/3
    DualTrits b(1, 1);  // 3
    DualTrits result = a * b;
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 1u);  // 1/3 * 3 = 1
}

TEST(DualTritsMultiplication, MinusOneThirdTimesOne) {
    DualTrits a(2, 2);  // -1/3
    DualTrits b(0, 1);  // 1
    DualTrits result = a * b;
    EXPECT_EQ(result.getExponent(), 2u);
    EXPECT_EQ(result.getDirection(), 2u);  // -1/3 * 1 = -1/3
}

TEST(DualTritsMultiplication, MinusOneThirdTimesMinusOne) {
    DualTrits a(2, 2);  // -1/3
    DualTrits b(0, 2);  // -1
    DualTrits result = a * b;
    EXPECT_EQ(result.getExponent(), 2u);
    EXPECT_EQ(result.getDirection(), 1u);  // -1/3 * -1 = 1/3
}

TEST(DualTritsMultiplication, OneThirdTimesMinusThree) {
    DualTrits a(2, 1);  // 1/3
    DualTrits b(1, 2);  // -3
    DualTrits result = a * b;
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 2u);  // 1/3 * -3 = -1
}

// =============================
// Overflow Tests (to infinity)
// =============================

TEST(DualTritsMultiplication, ThreeTimesThree) {
    DualTrits a(1, 1);  // 3
    DualTrits b(1, 1);  // 3
    DualTrits result = a * b;
    // 3 * 3 = 9, overflow to +inf
    EXPECT_EQ(result.getExponent(), 1u);
    EXPECT_EQ(result.getDirection(), 0u);  // +inf
    EXPECT_TRUE(result.isSpecial());
}

TEST(DualTritsMultiplication, MinusThreeTimesMinusThree) {
    DualTrits a(1, 2);  // -3
    DualTrits b(1, 2);  // -3
    DualTrits result = a * b;
    // -3 * -3 = 9, overflow to +inf
    EXPECT_EQ(result.getExponent(), 1u);
    EXPECT_EQ(result.getDirection(), 0u);  // +inf
    EXPECT_TRUE(result.isSpecial());
}

TEST(DualTritsMultiplication, ThreeTimesMinusThree) {
    DualTrits a(1, 1);  // 3
    DualTrits b(1, 2);  // -3
    DualTrits result = a * b;
    // 3 * -3 = -9, overflow to -inf
    EXPECT_EQ(result.getExponent(), 2u);
    EXPECT_EQ(result.getDirection(), 0u);  // -inf
    EXPECT_TRUE(result.isSpecial());
}

// =============================
// Underflow Tests (to zero)
// =============================

TEST(DualTritsMultiplication, OneThirdTimesOneThird) {
    DualTrits a(2, 1);  // 1/3
    DualTrits b(2, 1);  // 1/3
    DualTrits result = a * b;
    // 1/3 * 1/3 = 1/9, underflow to 0
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 0u);  // 0
}

TEST(DualTritsMultiplication, MinusOneThirdTimesOneThird) {
    DualTrits a(2, 2);  // -1/3
    DualTrits b(2, 1);  // 1/3
    DualTrits result = a * b;
    // -1/3 * 1/3 = -1/9, underflow to 0
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 0u);  // 0
}

TEST(DualTritsMultiplication, MinusOneThirdTimesMinusOneThird) {
    DualTrits a(2, 2);  // -1/3
    DualTrits b(2, 2);  // -1/3
    DualTrits result = a * b;
    // -1/3 * -1/3 = 1/9, underflow to 0
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 0u);  // 0
}

// =============================
// Special Value Tests
// =============================

TEST(DualTritsMultiplication, InfTimesOne) {
    DualTrits a(1, 0);  // inf
    DualTrits b(0, 1);  // 1
    DualTrits result = a * b;
    // inf * 1 = inf
    EXPECT_EQ(result.getExponent(), 1u);
    EXPECT_TRUE(result.isSpecial());
}

TEST(DualTritsMultiplication, InfTimesMinusOne) {
    DualTrits a(1, 0);  // inf
    DualTrits b(0, 2);  // -1
    DualTrits result = a * b;
    // inf * -1 = -inf (or some representation of it)
    EXPECT_TRUE(result.isSpecial());
}

TEST(DualTritsMultiplication, MinusInfTimesOne) {
    DualTrits a(2, 0);  // -inf
    DualTrits b(0, 1);  // 1
    DualTrits result = a * b;
    // -inf * 1 = -inf
    EXPECT_EQ(result.getExponent(), 2u);
    EXPECT_TRUE(result.isSpecial());
}

// =============================
// Commutativity Tests
// =============================

TEST(DualTritsProperties, MultiplicationCommutative) {
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
            DualTrits ab = a * b;
            DualTrits ba = b * a;
            EXPECT_EQ(ab.getExponent(), ba.getExponent()) 
                << "a=" << a.getExponent() << "," << a.getDirection()
                << " b=" << b.getExponent() << "," << b.getDirection();
            EXPECT_EQ(ab.getDirection(), ba.getDirection())
                << "a=" << a.getExponent() << "," << a.getDirection()
                << " b=" << b.getExponent() << "," << b.getDirection();
        }
    }
}

// =============================
// Comprehensive Tests
// =============================

TEST(DualTritsMultiplication, AllFiniteValuesCombinations) {
    // Test all non-special value combinations
    struct TestCase {
        int exp_a, dir_a;
        int exp_b, dir_b;
        const char* desc;
    };
    
    std::vector<TestCase> cases = {
        {0, 1, 0, 1, "1 * 1 = 1"},
        {0, 1, 0, 2, "1 * -1 = -1"},
        {0, 1, 1, 1, "1 * 3 = 3"},
        {0, 1, 2, 1, "1 * 1/3 = 1/3"},
        {1, 1, 2, 1, "3 * 1/3 = 1"},
        {0, 2, 0, 2, "-1 * -1 = 1"},
        {1, 1, 0, 2, "3 * -1 = -3"},
        {2, 1, 2, 1, "1/3 * 1/3 = 1/9 -> 0"},
        {1, 1, 1, 1, "3 * 3 = 9 -> inf"},
    };
    
    for (const auto& tc : cases) {
        DualTrits a(tc.exp_a, tc.dir_a);
        DualTrits b(tc.exp_b, tc.dir_b);
        
        // Just verify operations don't crash and produce valid DualTrits
        DualTrits result = a * b;
        
        EXPECT_LE(result.getExponent(), 2u) << tc.desc;
        EXPECT_LE(result.getDirection(), 2u) << tc.desc;
    }
}

// =============================
// Identity Tests
// =============================

TEST(DualTritsMultiplication, MultiplyByOne) {
    std::vector<DualTrits> values = {
        DualTrits(0, 1),  // 1
        DualTrits(0, 2),  // -1
        DualTrits(2, 1),  // 1/3
        DualTrits(2, 2),  // -1/3
        DualTrits(1, 1),  // 3
        DualTrits(1, 2),  // -3
    };
    
    DualTrits one(0, 1);  // 1
    
    for (const auto& val : values) {
        DualTrits result = val * one;
        EXPECT_EQ(result.getExponent(), val.getExponent()) 
            << "val=" << val.getExponent() << "," << val.getDirection();
        EXPECT_EQ(result.getDirection(), val.getDirection())
            << "val=" << val.getExponent() << "," << val.getDirection();
    }
}

TEST(DualTritsMultiplication, MultiplyByZero) {
    std::vector<DualTrits> values = {
        DualTrits(0, 1),  // 1
        DualTrits(0, 2),  // -1
        DualTrits(2, 1),  // 1/3
        DualTrits(2, 2),  // -1/3
        DualTrits(1, 1),  // 3
        DualTrits(1, 2),  // -3
    };
    
    DualTrits zero(0, 0);  // 0
    
    for (const auto& val : values) {
        DualTrits result = val * zero;
        EXPECT_EQ(result.getExponent(), 0u) 
            << "val=" << val.getExponent() << "," << val.getDirection();
        EXPECT_EQ(result.getDirection(), 0u)
            << "val=" << val.getExponent() << "," << val.getDirection();
    }
}
