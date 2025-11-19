#include <gtest/gtest.h>
#include "common/DualTrits.hpp"
#include <stdexcept>

/*
 * DualTrits division tests
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
// Basic Division Tests
// =============================

TEST(DualTritsDivision, OneDividedByOne) {
    DualTrits a(0, 1);  // 1
    DualTrits b(0, 1);  // 1
    DualTrits result = a / b;
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 1u);  // 1 / 1 = 1
}

TEST(DualTritsDivision, ThreeDividedByOne) {
    DualTrits a(1, 1);  // 3
    DualTrits b(0, 1);  // 1
    DualTrits result = a / b;
    EXPECT_EQ(result.getExponent(), 1u);
    EXPECT_EQ(result.getDirection(), 1u);  // 3 / 1 = 3
}

TEST(DualTritsDivision, OneDividedByThree) {
    DualTrits a(0, 1);  // 1
    DualTrits b(1, 1);  // 3
    DualTrits result = a / b;
    EXPECT_EQ(result.getExponent(), 2u);
    EXPECT_EQ(result.getDirection(), 1u);  // 1 / 3 = 1/3
}

TEST(DualTritsDivision, ThreeDividedByThree) {
    DualTrits a(1, 1);  // 3
    DualTrits b(1, 1);  // 3
    DualTrits result = a / b;
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 1u);  // 3 / 3 = 1
}

TEST(DualTritsDivision, MinusOneDividedByOne) {
    DualTrits a(0, 2);  // -1
    DualTrits b(0, 1);  // 1
    DualTrits result = a / b;
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 2u);  // -1 / 1 = -1
}

TEST(DualTritsDivision, OneDividedByMinusOne) {
    DualTrits a(0, 1);  // 1
    DualTrits b(0, 2);  // -1
    DualTrits result = a / b;
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 2u);  // 1 / -1 = -1
}

TEST(DualTritsDivision, MinusOneDividedByMinusOne) {
    DualTrits a(0, 2);  // -1
    DualTrits b(0, 2);  // -1
    DualTrits result = a / b;
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 1u);  // -1 / -1 = 1
}

TEST(DualTritsDivision, ThreeDividedByMinusOne) {
    DualTrits a(1, 1);  // 3
    DualTrits b(0, 2);  // -1
    DualTrits result = a / b;
    EXPECT_EQ(result.getExponent(), 1u);
    EXPECT_EQ(result.getDirection(), 2u);  // 3 / -1 = -3
}

TEST(DualTritsDivision, MinusThreeDividedByOne) {
    DualTrits a(1, 2);  // -3
    DualTrits b(0, 1);  // 1
    DualTrits result = a / b;
    EXPECT_EQ(result.getExponent(), 1u);
    EXPECT_EQ(result.getDirection(), 2u);  // -3 / 1 = -3
}

TEST(DualTritsDivision, MinusThreeDividedByMinusOne) {
    DualTrits a(1, 2);  // -3
    DualTrits b(0, 2);  // -1
    DualTrits result = a / b;
    EXPECT_EQ(result.getExponent(), 1u);
    EXPECT_EQ(result.getDirection(), 1u);  // -3 / -1 = 3
}

// =============================
// Fraction Division Tests
// =============================

TEST(DualTritsDivision, OneThirdDividedByOne) {
    DualTrits a(2, 1);  // 1/3
    DualTrits b(0, 1);  // 1
    DualTrits result = a / b;
    EXPECT_EQ(result.getExponent(), 2u);
    EXPECT_EQ(result.getDirection(), 1u);  // 1/3 / 1 = 1/3
}

TEST(DualTritsDivision, OneThirdDividedByOneThird) {
    DualTrits a(2, 1);  // 1/3
    DualTrits b(2, 1);  // 1/3
    DualTrits result = a / b;
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 1u);  // 1/3 / 1/3 = 1
}

TEST(DualTritsDivision, OneDividedByOneThird) {
    DualTrits a(0, 1);  // 1
    DualTrits b(2, 1);  // 1/3
    DualTrits result = a / b;
    EXPECT_EQ(result.getExponent(), 1u);
    EXPECT_EQ(result.getDirection(), 1u);  // 1 / 1/3 = 3
}

TEST(DualTritsDivision, MinusOneThirdDividedByOne) {
    DualTrits a(2, 2);  // -1/3
    DualTrits b(0, 1);  // 1
    DualTrits result = a / b;
    EXPECT_EQ(result.getExponent(), 2u);
    EXPECT_EQ(result.getDirection(), 2u);  // -1/3 / 1 = -1/3
}

TEST(DualTritsDivision, MinusOneThirdDividedByMinusOne) {
    DualTrits a(2, 2);  // -1/3
    DualTrits b(0, 2);  // -1
    DualTrits result = a / b;
    EXPECT_EQ(result.getExponent(), 2u);
    EXPECT_EQ(result.getDirection(), 1u);  // -1/3 / -1 = 1/3
}

TEST(DualTritsDivision, MinusThreeDividedByThree) {
    DualTrits a(1, 2);  // -3
    DualTrits b(1, 1);  // 3
    DualTrits result = a / b;
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 2u);  // -3 / 3 = -1
}

// =============================
// Overflow Tests (to infinity)
// =============================

TEST(DualTritsDivision, ThreeDividedByOneThird) {
    DualTrits a(1, 1);  // 3
    DualTrits b(2, 1);  // 1/3
    DualTrits result = a / b;
    // 3 / 1/3 = 9, overflow to +inf
    EXPECT_EQ(result.getExponent(), 1u);
    EXPECT_EQ(result.getDirection(), 0u);  // +inf
    EXPECT_TRUE(result.isSpecial());
}

TEST(DualTritsDivision, MinusThreeDividedByMinusOneThird) {
    DualTrits a(1, 2);  // -3
    DualTrits b(2, 2);  // -1/3
    DualTrits result = a / b;
    // -3 / -1/3 = 9, overflow to +inf
    EXPECT_EQ(result.getExponent(), 1u);
    EXPECT_EQ(result.getDirection(), 0u);  // +inf
    EXPECT_TRUE(result.isSpecial());
}

TEST(DualTritsDivision, ThreeDividedByMinusOneThird) {
    DualTrits a(1, 1);  // 3
    DualTrits b(2, 2);  // -1/3
    DualTrits result = a / b;
    // 3 / -1/3 = -9, overflow to -inf
    EXPECT_EQ(result.getExponent(), 2u);
    EXPECT_EQ(result.getDirection(), 0u);  // -inf
    EXPECT_TRUE(result.isSpecial());
}

TEST(DualTritsDivision, MinusThreeDividedByOneThird) {
    DualTrits a(1, 2);  // -3
    DualTrits b(2, 1);  // 1/3
    DualTrits result = a / b;
    // -3 / 1/3 = -9, overflow to -inf
    EXPECT_EQ(result.getExponent(), 2u);
    EXPECT_EQ(result.getDirection(), 0u);  // -inf
    EXPECT_TRUE(result.isSpecial());
}

// =============================
// Underflow Tests (to zero)
// =============================

TEST(DualTritsDivision, OneThirdDividedByThree) {
    DualTrits a(2, 1);  // 1/3
    DualTrits b(1, 1);  // 3
    DualTrits result = a / b;
    // 1/3 / 3 = 1/9, underflow to 0
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 0u);  // 0
}

TEST(DualTritsDivision, MinusOneThirdDividedByThree) {
    DualTrits a(2, 2);  // -1/3
    DualTrits b(1, 1);  // 3
    DualTrits result = a / b;
    // -1/3 / 3 = -1/9, underflow to 0
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 0u);  // 0
}

TEST(DualTritsDivision, OneThirdDividedByMinusThree) {
    DualTrits a(2, 1);  // 1/3
    DualTrits b(1, 2);  // -3
    DualTrits result = a / b;
    // 1/3 / -3 = -1/9, underflow to 0
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 0u);  // 0
}

// =============================
// Zero Division Tests
// =============================

TEST(DualTritsDivision, ZeroDividedByOne) {
    DualTrits a(0, 0);  // 0
    DualTrits b(0, 1);  // 1
    DualTrits result = a / b;
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 0u);  // 0 / 1 = 0
}

TEST(DualTritsDivision, ZeroDividedByThree) {
    DualTrits a(0, 0);  // 0
    DualTrits b(1, 1);  // 3
    DualTrits result = a / b;
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 0u);  // 0 / 3 = 0
}

TEST(DualTritsDivision, ZeroDividedByMinusOne) {
    DualTrits a(0, 0);  // 0
    DualTrits b(0, 2);  // -1
    DualTrits result = a / b;
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 0u);  // 0 / -1 = 0
}

TEST(DualTritsDivision, ZeroDividedByOneThird) {
    DualTrits a(0, 0);  // 0
    DualTrits b(2, 1);  // 1/3
    DualTrits result = a / b;
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 0u);  // 0 / 1/3 = 0
}

// =============================
// Exception Tests (Division by Zero)
// =============================

TEST(DualTritsDivision, OneDividedByZeroThrows) {
    DualTrits a(0, 1);  // 1
    DualTrits b(0, 0);  // 0
    EXPECT_THROW({
        DualTrits result = a / b;
    }, std::domain_error);
}

TEST(DualTritsDivision, ThreeDividedByZeroThrows) {
    DualTrits a(1, 1);  // 3
    DualTrits b(0, 0);  // 0
    EXPECT_THROW({
        DualTrits result = a / b;
    }, std::domain_error);
}

TEST(DualTritsDivision, MinusOneDividedByZeroThrows) {
    DualTrits a(0, 2);  // -1
    DualTrits b(0, 0);  // 0
    EXPECT_THROW({
        DualTrits result = a / b;
    }, std::domain_error);
}

TEST(DualTritsDivision, ZeroDividedByZeroThrows) {
    DualTrits a(0, 0);  // 0
    DualTrits b(0, 0);  // 0
    EXPECT_THROW({
        DualTrits result = a / b;
    }, std::domain_error);
}

TEST(DualTritsDivision, OneThirdDividedByZeroThrows) {
    DualTrits a(2, 1);  // 1/3
    DualTrits b(0, 0);  // 0
    EXPECT_THROW({
        DualTrits result = a / b;
    }, std::domain_error);
}

// =============================
// Special Value Tests
// =============================

TEST(DualTritsDivision, InfDividedByOne) {
    DualTrits a(1, 0);  // inf
    DualTrits b(0, 1);  // 1
    DualTrits result = a / b;
    // inf / 1 = inf
    EXPECT_EQ(result.getExponent(), 1u);
    EXPECT_EQ(result.getDirection(), 0u);
    EXPECT_TRUE(result.isSpecial());
}

TEST(DualTritsDivision, InfDividedByMinusOne) {
    DualTrits a(1, 0);  // inf
    DualTrits b(0, 2);  // -1
    DualTrits result = a / b;
    // inf / -1 = -inf
    EXPECT_EQ(result.getExponent(), 2u);
    EXPECT_EQ(result.getDirection(), 0u);
    EXPECT_TRUE(result.isSpecial());
}

TEST(DualTritsDivision, MinusInfDividedByOne) {
    DualTrits a(2, 0);  // -inf
    DualTrits b(0, 1);  // 1
    DualTrits result = a / b;
    // -inf / 1 = -inf
    EXPECT_EQ(result.getExponent(), 2u);
    EXPECT_EQ(result.getDirection(), 0u);
    EXPECT_TRUE(result.isSpecial());
}

TEST(DualTritsDivision, MinusInfDividedByMinusOne) {
    DualTrits a(2, 0);  // -inf
    DualTrits b(0, 2);  // -1
    DualTrits result = a / b;
    // -inf / -1 = inf
    EXPECT_EQ(result.getExponent(), 1u);
    EXPECT_EQ(result.getDirection(), 0u);
    EXPECT_TRUE(result.isSpecial());
}

TEST(DualTritsDivision, InfDividedByThree) {
    DualTrits a(1, 0);  // inf
    DualTrits b(1, 1);  // 3
    DualTrits result = a / b;
    // inf / 3 = inf
    EXPECT_EQ(result.getExponent(), 1u);
    EXPECT_EQ(result.getDirection(), 0u);
    EXPECT_TRUE(result.isSpecial());
}

TEST(DualTritsDivision, OneDividedByInf) {
    DualTrits a(0, 1);  // 1
    DualTrits b(1, 0);  // inf
    DualTrits result = a / b;
    // 1 / inf = 0
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 0u);
}

TEST(DualTritsDivision, ThreeDividedByInf) {
    DualTrits a(1, 1);  // 3
    DualTrits b(1, 0);  // inf
    DualTrits result = a / b;
    // 3 / inf = 0
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 0u);
}

TEST(DualTritsDivision, MinusOneDividedByInf) {
    DualTrits a(0, 2);  // -1
    DualTrits b(1, 0);  // inf
    DualTrits result = a / b;
    // -1 / inf = 0
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 0u);
}

TEST(DualTritsDivision, InfDividedByInf) {
    DualTrits a(1, 0);  // inf
    DualTrits b(1, 0);  // inf
    DualTrits result = a / b;
    // inf / inf = 1 (by convention)
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 1u);
}

TEST(DualTritsDivision, MinusInfDividedByInf) {
    DualTrits a(2, 0);  // -inf
    DualTrits b(1, 0);  // inf
    DualTrits result = a / b;
    // -inf / inf = -1 (by convention)
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 2u);
}

TEST(DualTritsDivision, InfDividedByMinusInf) {
    DualTrits a(1, 0);  // inf
    DualTrits b(2, 0);  // -inf
    DualTrits result = a / b;
    // inf / -inf = -1 (by convention)
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 2u);
}

TEST(DualTritsDivision, MinusInfDividedByMinusInf) {
    DualTrits a(2, 0);  // -inf
    DualTrits b(2, 0);  // -inf
    DualTrits result = a / b;
    // -inf / -inf = 1 (by convention)
    EXPECT_EQ(result.getExponent(), 0u);
    EXPECT_EQ(result.getDirection(), 1u);
}

// =============================
// Identity and Inverse Tests
// =============================

TEST(DualTritsDivision, DivideByOne) {
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
        DualTrits result = val / one;
        EXPECT_EQ(result.getExponent(), val.getExponent()) 
            << "val=" << val.getExponent() << "," << val.getDirection();
        EXPECT_EQ(result.getDirection(), val.getDirection())
            << "val=" << val.getExponent() << "," << val.getDirection();
    }
}

TEST(DualTritsDivision, MultiplyDivideInverse) {
    // Test that (a * b) / b = a for representable values
    std::vector<DualTrits> values = {
        DualTrits(0, 1),  // 1
        DualTrits(0, 2),  // -1
        DualTrits(2, 1),  // 1/3
        DualTrits(2, 2),  // -1/3
        DualTrits(1, 1),  // 3
        DualTrits(1, 2),  // -3
    };
    
    for (const auto& a : values) {
        for (const auto& b : values) {
            DualTrits product = a * b;
            // Only test if product didn't overflow/underflow
            if (!product.isSpecial() && product.getDirection() != 0) {
                DualTrits result = product / b;
                EXPECT_EQ(result.getExponent(), a.getExponent())
                    << "a=" << a.getExponent() << "," << a.getDirection()
                    << " b=" << b.getExponent() << "," << b.getDirection();
                EXPECT_EQ(result.getDirection(), a.getDirection())
                    << "a=" << a.getExponent() << "," << a.getDirection()
                    << " b=" << b.getExponent() << "," << b.getDirection();
            }
        }
    }
}

// =============================
// Comprehensive Tests
// =============================

TEST(DualTritsDivision, AllFiniteValuesCombinations) {
    // Test all non-special, non-zero value combinations
    struct TestCase {
        int exp_a, dir_a;
        int exp_b, dir_b;
        const char* desc;
    };
    
    std::vector<TestCase> cases = {
        {0, 1, 0, 1, "1 / 1 = 1"},
        {0, 1, 0, 2, "1 / -1 = -1"},
        {0, 1, 1, 1, "1 / 3 = 1/3"},
        {1, 1, 0, 1, "3 / 1 = 3"},
        {1, 1, 2, 1, "3 / 1/3 = 9 -> inf"},
        {0, 2, 0, 2, "-1 / -1 = 1"},
        {1, 1, 0, 2, "3 / -1 = -3"},
        {2, 1, 1, 1, "1/3 / 3 = 1/9 -> 0"},
        {1, 1, 1, 1, "3 / 3 = 1"},
    };
    
    for (const auto& tc : cases) {
        DualTrits a(tc.exp_a, tc.dir_a);
        DualTrits b(tc.exp_b, tc.dir_b);
        
        // Just verify operations don't crash and produce valid DualTrits
        DualTrits result = a / b;
        
        EXPECT_LE(result.getExponent(), 2u) << tc.desc;
        EXPECT_LE(result.getDirection(), 2u) << tc.desc;
    }
}
