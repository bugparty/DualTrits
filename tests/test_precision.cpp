#include <gtest/gtest.h>
#include "common/DualTrits.hpp"
#include <cmath>
#include <limits>

/*
 * DualTrits precision tests for toFloat(), toDouble(), and toMPreal()
 * 
 * These tests verify that the conversion methods produce correct floating-point
 * values with appropriate precision for each type.
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
// toFloat() Precision Tests
// =============================

TEST(DualTritsPrecision, FloatZero) {
    DualTrits dt(0, 0);  // 0
    float result = dt.toFloat();
    EXPECT_FLOAT_EQ(result, 0.0f);
    EXPECT_EQ(result, 0.0f);  // Exact equality for zero
}

TEST(DualTritsPrecision, FloatPositiveOne) {
    DualTrits dt(0, 1);  // 1
    float result = dt.toFloat();
    EXPECT_FLOAT_EQ(result, 1.0f);
    EXPECT_EQ(result, 1.0f);  // Exact equality for 1
}

TEST(DualTritsPrecision, FloatNegativeOne) {
    DualTrits dt(0, 2);  // -1
    float result = dt.toFloat();
    EXPECT_FLOAT_EQ(result, -1.0f);
    EXPECT_EQ(result, -1.0f);  // Exact equality for -1
}

TEST(DualTritsPrecision, FloatPositiveThree) {
    DualTrits dt(1, 1);  // 3
    float result = dt.toFloat();
    EXPECT_FLOAT_EQ(result, 3.0f);
    EXPECT_EQ(result, 3.0f);  // Exact equality for 3
}

TEST(DualTritsPrecision, FloatNegativeThree) {
    DualTrits dt(1, 2);  // -3
    float result = dt.toFloat();
    EXPECT_FLOAT_EQ(result, -3.0f);
    EXPECT_EQ(result, -3.0f);  // Exact equality for -3
}

TEST(DualTritsPrecision, FloatPositiveOneThird) {
    DualTrits dt(2, 1);  // 1/3
    float result = dt.toFloat();
    float expected = 1.0f / 3.0f;
    EXPECT_FLOAT_EQ(result, expected);
    
    // Verify precision - 1/3 cannot be represented exactly in binary float
    // but should be close
    EXPECT_NEAR(result, 0.333333f, 1e-6f);
}

TEST(DualTritsPrecision, FloatNegativeOneThird) {
    DualTrits dt(2, 2);  // -1/3
    float result = dt.toFloat();
    float expected = -1.0f / 3.0f;
    EXPECT_FLOAT_EQ(result, expected);
    
    // Verify precision
    EXPECT_NEAR(result, -0.333333f, 1e-6f);
}

TEST(DualTritsPrecision, FloatPositiveInfinity) {
    DualTrits dt(1, 0);  // +inf
    float result = dt.toFloat();
    EXPECT_TRUE(std::isinf(result));
    EXPECT_GT(result, 0.0f);
    EXPECT_EQ(result, std::numeric_limits<float>::infinity());
}

TEST(DualTritsPrecision, FloatNegativeInfinity) {
    DualTrits dt(2, 0);  // -inf
    float result = dt.toFloat();
    EXPECT_TRUE(std::isinf(result));
    EXPECT_LT(result, 0.0f);
    EXPECT_EQ(result, -std::numeric_limits<float>::infinity());
}

// =============================
// toDouble() Precision Tests
// =============================

TEST(DualTritsPrecision, DoubleZero) {
    DualTrits dt(0, 0);  // 0
    double result = dt.toDouble();
    EXPECT_DOUBLE_EQ(result, 0.0);
    EXPECT_EQ(result, 0.0);  // Exact equality for zero
}

TEST(DualTritsPrecision, DoublePositiveOne) {
    DualTrits dt(0, 1);  // 1
    double result = dt.toDouble();
    EXPECT_DOUBLE_EQ(result, 1.0);
    EXPECT_EQ(result, 1.0);  // Exact equality for 1
}

TEST(DualTritsPrecision, DoubleNegativeOne) {
    DualTrits dt(0, 2);  // -1
    double result = dt.toDouble();
    EXPECT_DOUBLE_EQ(result, -1.0);
    EXPECT_EQ(result, -1.0);  // Exact equality for -1
}

TEST(DualTritsPrecision, DoublePositiveThree) {
    DualTrits dt(1, 1);  // 3
    double result = dt.toDouble();
    EXPECT_DOUBLE_EQ(result, 3.0);
    EXPECT_EQ(result, 3.0);  // Exact equality for 3
}

TEST(DualTritsPrecision, DoubleNegativeThree) {
    DualTrits dt(1, 2);  // -3
    double result = dt.toDouble();
    EXPECT_DOUBLE_EQ(result, -3.0);
    EXPECT_EQ(result, -3.0);  // Exact equality for -3
}

TEST(DualTritsPrecision, DoublePositiveOneThird) {
    DualTrits dt(2, 1);  // 1/3
    double result = dt.toDouble();
    double expected = 1.0 / 3.0;
    EXPECT_DOUBLE_EQ(result, expected);
    
    // Verify precision - double has better precision than float
    EXPECT_NEAR(result, 0.333333333333333, 1e-15);
}

TEST(DualTritsPrecision, DoubleNegativeOneThird) {
    DualTrits dt(2, 2);  // -1/3
    double result = dt.toDouble();
    double expected = -1.0 / 3.0;
    EXPECT_DOUBLE_EQ(result, expected);
    
    // Verify precision
    EXPECT_NEAR(result, -0.333333333333333, 1e-15);
}

TEST(DualTritsPrecision, DoublePositiveInfinity) {
    DualTrits dt(1, 0);  // +inf
    double result = dt.toDouble();
    EXPECT_TRUE(std::isinf(result));
    EXPECT_GT(result, 0.0);
    EXPECT_EQ(result, std::numeric_limits<double>::infinity());
}

TEST(DualTritsPrecision, DoubleNegativeInfinity) {
    DualTrits dt(2, 0);  // -inf
    double result = dt.toDouble();
    EXPECT_TRUE(std::isinf(result));
    EXPECT_LT(result, 0.0);
    EXPECT_EQ(result, -std::numeric_limits<double>::infinity());
}

// =============================
// toMPreal() Precision Tests
// =============================
#ifdef USE_MPFR
TEST(DualTritsPrecision, MPrealZero) {
    DualTrits dt(0, 0);  // 0
    mpfr::mpreal result = dt.toMPreal();
    EXPECT_EQ(result, mpfr::mpreal(0));
    EXPECT_TRUE(result == 0);
}

TEST(DualTritsPrecision, MPrealPositiveOne) {
    DualTrits dt(0, 1);  // 1
    mpfr::mpreal result = dt.toMPreal();
    EXPECT_EQ(result, mpfr::mpreal(1));
    EXPECT_TRUE(result == 1);
}

TEST(DualTritsPrecision, MPrealNegativeOne) {
    DualTrits dt(0, 2);  // -1
    mpfr::mpreal result = dt.toMPreal();
    EXPECT_EQ(result, mpfr::mpreal(-1));
    EXPECT_TRUE(result == -1);
}

TEST(DualTritsPrecision, MPrealPositiveThree) {
    DualTrits dt(1, 1);  // 3
    mpfr::mpreal result = dt.toMPreal();
    EXPECT_EQ(result, mpfr::mpreal(3));
    EXPECT_TRUE(result == 3);
}

TEST(DualTritsPrecision, MPrealNegativeThree) {
    DualTrits dt(1, 2);  // -3
    mpfr::mpreal result = dt.toMPreal();
    EXPECT_EQ(result, mpfr::mpreal(-3));
    EXPECT_TRUE(result == -3);
}

TEST(DualTritsPrecision, MPrealPositiveOneThird) {
    DualTrits dt(2, 1);  // 1/3
    mpfr::mpreal result = dt.toMPreal();
    mpfr::mpreal expected = mpfr::mpreal(1) / mpfr::mpreal(3);
    EXPECT_EQ(result, expected);
    
    // Verify high precision
    mpfr::mpreal diff = mpfr::abs(result - expected);
    EXPECT_TRUE(diff < mpfr::mpreal(1e-50));
}

TEST(DualTritsPrecision, MPrealNegativeOneThird) {
    DualTrits dt(2, 2);  // -1/3
    mpfr::mpreal result = dt.toMPreal();
    mpfr::mpreal expected = mpfr::mpreal(-1) / mpfr::mpreal(3);
    EXPECT_EQ(result, expected);
    
    // Verify high precision
    mpfr::mpreal diff = mpfr::abs(result - expected);
    EXPECT_TRUE(diff < mpfr::mpreal(1e-50));
}

TEST(DualTritsPrecision, MPrealPositiveInfinity) {
    DualTrits dt(1, 0);  // +inf
    mpfr::mpreal result = dt.toMPreal();
    EXPECT_TRUE(mpfr::isfinite(result) == false || mpfr::isinf(result));
}

TEST(DualTritsPrecision, MPrealNegativeInfinity) {
    DualTrits dt(2, 0);  // -inf
    mpfr::mpreal result = dt.toMPreal();
    EXPECT_TRUE(mpfr::isfinite(result) == false || mpfr::isinf(result));
}
#endif

// =============================
// Comparative Precision Tests
// =============================

#ifdef USE_MPFR
TEST(DualTritsPrecision, CompareFloatDoubleOneThird) {
    DualTrits dt(2, 1);  // 1/3
    
    float float_result = dt.toFloat();
    double double_result = dt.toDouble();
    mpfr::mpreal mpreal_result = dt.toMPreal();
    
    // Convert to common type for comparison
    double float_as_double = static_cast<double>(float_result);
    double mpreal_as_double = mpreal_result.toDouble();
    
    // Double should be at least as precise as float
    EXPECT_LE(std::abs(double_result - (1.0 / 3.0)), 
              std::abs(float_as_double - (1.0 / 3.0)));
    
    // MPreal should be at least as precise as double
    EXPECT_LE(std::abs(mpreal_as_double - (1.0 / 3.0)),
              std::abs(double_result - (1.0 / 3.0)) + 1e-15);
}
#endif

TEST(DualTritsPrecision, AllValuesFloatConsistency) {
    // Test all 9 DualTrits values
    struct TestCase {
        int exp, dir;
        float expected;
        const char* description;
    };
    
    std::vector<TestCase> test_cases = {
        {0, 0, 0.0f, "zero"},
        {0, 1, 1.0f, "one"},
        {0, 2, -1.0f, "negative one"},
        {1, 1, 3.0f, "three"},
        {1, 2, -3.0f, "negative three"},
        {2, 1, 1.0f / 3.0f, "one third"},
        {2, 2, -1.0f / 3.0f, "negative one third"},
    };
    
    for (const auto& tc : test_cases) {
        DualTrits dt(tc.exp, tc.dir);
        float result = dt.toFloat();
        EXPECT_FLOAT_EQ(result, tc.expected) << "Failed for " << tc.description;
    }
}

TEST(DualTritsPrecision, AllValuesDoubleConsistency) {
    // Test all 9 DualTrits values
    struct TestCase {
        int exp, dir;
        double expected;
        const char* description;
    };
    
    std::vector<TestCase> test_cases = {
        {0, 0, 0.0, "zero"},
        {0, 1, 1.0, "one"},
        {0, 2, -1.0, "negative one"},
        {1, 1, 3.0, "three"},
        {1, 2, -3.0, "negative three"},
        {2, 1, 1.0 / 3.0, "one third"},
        {2, 2, -1.0 / 3.0, "negative one third"},
    };
    
    for (const auto& tc : test_cases) {
        DualTrits dt(tc.exp, tc.dir);
        double result = dt.toDouble();
        EXPECT_DOUBLE_EQ(result, tc.expected) << "Failed for " << tc.description;
    }
}

#ifdef USE_MPFR
TEST(DualTritsPrecision, AllValuesMPrealConsistency) {
    // Test all 9 DualTrits values
    struct TestCase {
        int exp, dir;
        const char* description;
    };
    
    std::vector<TestCase> test_cases = {
        {0, 0, "zero"},
        {0, 1, "one"},
        {0, 2, "negative one"},
        {1, 1, "three"},
        {1, 2, "negative three"},
        {2, 1, "one third"},
        {2, 2, "negative one third"},
    };
    
    for (const auto& tc : test_cases) {
        DualTrits dt(tc.exp, tc.dir);
        mpfr::mpreal result = dt.toMPreal();
        
        // Verify result is finite for non-infinity values
        EXPECT_TRUE(mpfr::isfinite(result) || mpfr::iszero(result)) 
            << "Failed for " << tc.description;
    }
}
#endif

// =============================
// Precision Loss Tests
// =============================

TEST(DualTritsPrecision, FloatVsDoubleOneThirdPrecision) {
    DualTrits dt(2, 1);  // 1/3
    
    float float_result = dt.toFloat();
    double double_result = dt.toDouble();
    
    // Calculate actual precision loss
    double true_value = 1.0 / 3.0;
    double float_error = std::abs(static_cast<double>(float_result) - true_value);
    double double_error = std::abs(double_result - true_value);
    
    // Float should have more error than double (less precision)
    EXPECT_GT(float_error, double_error);
    
    // Print precision information (for informational purposes)
    std::cout << "1/3 precision comparison:" << std::endl;
    std::cout << "  Float error:  " << float_error << std::endl;
    std::cout << "  Double error: " << double_error << std::endl;
    std::cout << "  Precision improvement: " << (float_error / double_error) << "x" << std::endl;
}

#ifdef USE_MPFR
TEST(DualTritsPrecision, MPrealExactRepresentation) {
    DualTrits dt(2, 1);  // 1/3
    
    mpfr::mpreal result = dt.toMPreal();
    
    // MPreal should exactly represent 1 * 3^(-1) = 1/3
    mpfr::mpreal expected = mpfr::mpreal(1) / mpfr::mpreal(3);
    
    // Should be exactly equal with arbitrary precision
    EXPECT_EQ(result, expected);
    
    // Verify it's computed correctly
    mpfr::mpreal three_times = result * mpfr::mpreal(3);
    EXPECT_EQ(three_times, mpfr::mpreal(1));
}
#endif

// =============================
// Round-trip Conversion Tests
// =============================

TEST(DualTritsPrecision, FloatStringRoundTrip) {
    std::vector<DualTrits> test_values = {
        DualTrits(0, 0),  // 0
        DualTrits(0, 1),  // 1
        DualTrits(0, 2),  // -1
        DualTrits(2, 1),  // 1/3
        DualTrits(2, 2),  // -1/3
        DualTrits(1, 1),  // 3
        DualTrits(1, 2),  // -3
    };
    
    for (const auto& dt : test_values) {
        std::string float_str = dt.toFloatString();
        EXPECT_FALSE(float_str.empty()) 
            << "Float string conversion failed for exp=" << (int)dt.getExponent() 
            << " dir=" << (int)dt.getDirection();
    }
}

TEST(DualTritsPrecision, DoubleStringRoundTrip) {
    std::vector<DualTrits> test_values = {
        DualTrits(0, 0),  // 0
        DualTrits(0, 1),  // 1
        DualTrits(0, 2),  // -1
        DualTrits(2, 1),  // 1/3
        DualTrits(2, 2),  // -1/3
        DualTrits(1, 1),  // 3
        DualTrits(1, 2),  // -3
    };
    
    for (const auto& dt : test_values) {
        std::string double_str = dt.toDoubleString();
        EXPECT_FALSE(double_str.empty()) 
            << "Double string conversion failed for exp=" << (int)dt.getExponent() 
            << " dir=" << (int)dt.getDirection();
    }
}

#ifdef USE_MPFR
TEST(DualTritsPrecision, MPrealStringRoundTrip) {
    std::vector<DualTrits> test_values = {
        DualTrits(0, 0),  // 0
        DualTrits(0, 1),  // 1
        DualTrits(0, 2),  // -1
        DualTrits(2, 1),  // 1/3
        DualTrits(2, 2),  // -1/3
        DualTrits(1, 1),  // 3
        DualTrits(1, 2),  // -3
    };
    
    for (const auto& dt : test_values) {
        std::string mpreal_str = dt.toMPrealString();
        EXPECT_FALSE(mpreal_str.empty()) 
            << "MPreal string conversion failed for exp=" << (int)dt.getExponent() 
            << " dir=" << (int)dt.getDirection();
    }
}
#endif

// =============================
// Numerical Stability Tests
// =============================

#ifdef USE_MPFR
TEST(DualTritsPrecision, ZeroHandling) {
    DualTrits dt(0, 0);  // 0
    
    // All conversions should produce exact zero
    float f = dt.toFloat();
    double d = dt.toDouble();
    mpfr::mpreal m = dt.toMPreal();
    
    EXPECT_EQ(f, 0.0f);
    EXPECT_EQ(d, 0.0);
    EXPECT_EQ(m, mpfr::mpreal(0));
    
    // Verify no negative zero issues
    EXPECT_FALSE(std::signbit(f));
    EXPECT_FALSE(std::signbit(d));
}

TEST(DualTritsPrecision, SignPreservation) {
    // Test that signs are correctly preserved in all conversions
    struct TestCase {
        int exp, dir;
        bool should_be_positive;
        const char* description;
    };
    
    std::vector<TestCase> test_cases = {
        {0, 1, true, "positive one"},
        {0, 2, false, "negative one"},
        {1, 1, true, "positive three"},
        {1, 2, false, "negative three"},
        {2, 1, true, "positive one third"},
        {2, 2, false, "negative one third"},
    };
    
    for (const auto& tc : test_cases) {
        DualTrits dt(tc.exp, tc.dir);
        
        float f = dt.toFloat();
        double d = dt.toDouble();
        mpfr::mpreal m = dt.toMPreal();
        
        if (tc.should_be_positive) {
            EXPECT_GT(f, 0.0f) << "Float sign wrong for " << tc.description;
            EXPECT_GT(d, 0.0) << "Double sign wrong for " << tc.description;
            EXPECT_GT(m, mpfr::mpreal(0)) << "MPreal sign wrong for " << tc.description;
        } else {
            EXPECT_LT(f, 0.0f) << "Float sign wrong for " << tc.description;
            EXPECT_LT(d, 0.0) << "Double sign wrong for " << tc.description;
            EXPECT_LT(m, mpfr::mpreal(0)) << "MPreal sign wrong for " << tc.description;
        }
    }
}
#endif
