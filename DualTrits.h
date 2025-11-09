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
    typedef int16_t compute_t;
    static constexpr wide_t BASE = 3;
    /*
    our dual trits representation:
    exponent: 0 -> 0 (3^0)
              1 -> 1 (3^1)
              2 -> 2 (3^-1)
    direction: 0 -> 0 (0)
               1 -> 1 (1)
               2 -> 2 (-1)
    number range: -inf, -3,-1,-1/3,0,1/3,1,3,inf
    DualTrits(0,0) = 0
    DualTrits(1,0) = inf
    DualTrits(2,0) = -inf
    DualTrits(0,1) = 1
    DualTrits(0,2) = -1
    DualTrits(1,1) = 3
    DualTrits(1,2) = -3
    DualTrits(2,1) = 1/3
    DualTrits(2,2) = -1/3
    */
    constexpr DualTrits(int e = 0, wide_t d = 0) noexcept : exponent(e), direction(d) {}
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
    bool isSpecial() const noexcept {
        return (direction == 0 && exponent != 0);
    }
    compute_t mul3() const;
    DualTrits divide3(compute_t num) const;
    DualTrits round_mul3(compute_t num) const;
    // Swap operation
    void swap(DualTrits& other) noexcept;

    std::bitset<4> asBits() const noexcept;
    unsigned int asRawBits() const noexcept;
    std::bitset<4> asPackedBits() const noexcept;
    unsigned int asRawPackedBits() const noexcept;

    template <std::size_t Count, class UInt>
    friend constexpr UInt pack_dual_trits(DualTrits const* dual_trits);
    template <std::size_t Count, class UInt>
    friend constexpr void unpack_dual_trits(UInt packed, DualTrits* out) noexcept;

    // Non-member swap function
    friend void swap(DualTrits& a, DualTrits& b) noexcept {
        a.swap(b);
    }

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
    [[nodiscard]] static int constexpr compare_digit(wide_t l, wide_t r) noexcept {
        if (l == r) return 0;
        if (l == 2) return -1; // -1 < 0, 1
        if (r == 2) return 1;  // 0, 1 > -1
        //now both l and r are in {0,1}
        if (l < r) return -1;
        if (l > r) return 1;
        return 0;
    }
    template<typename T, wide_t BASE>
    [[nodiscard]] T constexpr pow_base(wide_t exp) const noexcept;

    unsigned int exponent : 2;
    unsigned int direction : 2;
};

#endif //PROJECT_FLOAT_DUALTRITS_H
