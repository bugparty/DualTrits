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
    constexpr DualTrits() noexcept : exponent(0), direction(0) {}
    constexpr explicit DualTrits(int e, wide_t d) noexcept : exponent(e), direction(d) {}
    // Accessors for testing and inspection
    [[nodiscard]] constexpr unsigned int getExponent() const noexcept { return exponent; }
    [[nodiscard]] constexpr unsigned int getDirection() const noexcept { return direction; }
    std::string toString() const;
    std::string toFancyString() const;

    float toFloat() const noexcept;
    double toDouble() const noexcept;
    mpfr::mpreal toMPreal() const noexcept;

    std::string toFloatString() const;
    std::string toDoubleString() const;
    std::string toMPrealString() const;

    DualTrits operator+(const DualTrits& other) const;
    DualTrits operator-(const DualTrits& other) const;
    DualTrits operator*(const DualTrits& other) const;
    DualTrits operator/(const DualTrits& other) const;

    std::bitset<4> asBits() const noexcept;
    unsigned int asRawBits() const noexcept;
    std::bitset<4> asPackedBits() const noexcept;
    unsigned int asRawPackedBits() const noexcept;

    template <std::size_t Count, class UInt>
    friend constexpr UInt pack_dual_trits(DualTrits const* dual_trits);
    template <std::size_t Count, class UInt>
    friend constexpr void unpack_dual_trits(UInt packed, DualTrits* out) noexcept;

private:
    template<typename T>
    [[nodiscard]] constexpr T to() const noexcept;

    template<typename T>
    [[nodiscard]] std::string toAsString() const;

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
    unsigned int direction : 2;
};

#endif //PROJECT_FLOAT_DUALTRITS_H
