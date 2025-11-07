//
// Created by bowman on 2025/10/22.
//

#ifndef PROJECT_FLOAT_DUALTRITS_H
#define PROJECT_FLOAT_DUALTRITS_H

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <ostream>
#include <sstream>
#include <bitset>

#include <mpreal.h>

class DualTrits {
public:
    typedef int8_t wide_t;
    static constexpr wide_t BASE = 3;

    constexpr explicit DualTrits(wide_t m = 0, int e = 0) noexcept : exponent(e), mantissa(m) {}

    std::string toString() const noexcept;
    std::string toFancyString() const noexcept;

    float toFloat() const noexcept;
    double toDouble() const noexcept;
    mpfr::mpreal toMPreal() const noexcept;

    std::string toFloatString();
    std::string toDoubleString();
    std::string toMPrealString();

    DualTrits operator+(const DualTrits& other) const;
    DualTrits operator-(const DualTrits& other) const;
    DualTrits operator*(const DualTrits& other) const;
    DualTrits operator/(const DualTrits& other) const;

    std::bitset<4> asBits() const noexcept;
    std::bitset<4> asPackedBits() const noexcept;

private:
    template<typename T>
    [[nodiscard]] constexpr T to() const noexcept;

    template<typename T>
    [[nodiscard]] std::string toAsString() const noexcept;

    [[nodiscard]] static wide_t constexpr reinterpt_digit(wide_t digit) noexcept {
        if (digit == 2) {
            return -1;
        }
        if (digit == 1) {
            return 1;
        }
        if (digit == 0) {
            return 0;
        }
        return 0;
    }

    template<typename T, wide_t BASE>
    [[nodiscard]] T constexpr pow_base(wide_t exp) const noexcept;

    unsigned int exponent : 2;
    unsigned int mantissa : 2;
};

#endif //PROJECT_FLOAT_DUALTRITS_H
