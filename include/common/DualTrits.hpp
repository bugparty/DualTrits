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
#include <stdexcept>

#ifdef USE_MPFR

#if __has_include(<mpreal.h>)

#include <mpreal.h>

#else

#include <3rdparty/mpreal.h>

#endif

#endif

#ifndef __device__
#define __host__
#define __device__
#endif

class DualTrits {
public:
    typedef int8_t wide_t;
    typedef int16_t compute_t;
    static constexpr wide_t BASE = 3;
    
    // Constants for positive and negative infinity
    static const DualTrits POSITIVE_INFINITY;
    static const DualTrits NEGATIVE_INFINITY;
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
    __host__ __device__
    constexpr DualTrits(int e = 0, wide_t d = 0) noexcept : storage((e << 2) | (d & 0b11)) {}
    // Accessors for testing and inspection
    __host__ __device__
    [[nodiscard]] constexpr int8_t getExponent() const noexcept { return (storage >> 2) & 0b11; }
    constexpr void setExponent(int8_t e) noexcept {
        storage = ( storage & 0b11 ) | ( (e & 0b11) << 2 );
    }
    __host__ __device__
    [[nodiscard]] constexpr int8_t getDirection() const noexcept { return storage & 0b11; }
    constexpr void setDirection(int8_t d) noexcept {
        storage = ( storage & 0b1100 ) | ( d & 0b11 );
    }
    std::string toString() const;
    std::string toFancyString() const;

    float toFloat() const noexcept;
    double toDouble() const noexcept;
#ifdef USE_MPFR
    mpfr::mpreal toMPreal() const noexcept;
#endif

    std::string toFloatString() const;
    std::string toDoubleString() const;
#ifdef USE_MPFR
    std::string toMPrealString() const;
#endif

    DualTrits operator+(const DualTrits& other) const;
    DualTrits operator-(const DualTrits& other) const;
    DualTrits operator*(const DualTrits& other) const;
    DualTrits operator/(const DualTrits& other) const;
    bool isSpecial() const noexcept {
        return (getDirection() == 0 && getExponent() != 0);
    }
    
    // Infinity checking functions
    bool isInfinity() const noexcept {
        return (getDirection() == 0 && getExponent() != 0);
    }
    
    bool isPositiveInfinity() const noexcept {
        return (getDirection() == 0 && getExponent() == 1);
    }
    
    bool isNegativeInfinity() const noexcept {
        return (getDirection() == 0 && getExponent() == 2);
    }
    
    compute_t mul3() const;
    static DualTrits divide3(compute_t num);
    DualTrits round_mul3(compute_t num) const;
    static DualTrits rounding_float(float num);
    static DualTrits construct_from_index(int index);
    
    // Helper functions for multiplication
    [[nodiscard]] static int reinterpt_exponent(wide_t exp) noexcept;
    [[nodiscard]] static wide_t encode_exponent(int exp_val) noexcept;
    [[nodiscard]] static wide_t encode_direction(int dir_val) noexcept;
    
    // Swap operation
    void swap(DualTrits& other) noexcept;

    std::bitset<4> asBits() const noexcept;
    unsigned int asRawBits() const noexcept;
    std::bitset<4> asPackedBits() const noexcept;
    __host__ __device__ unsigned int asRawPackedBits() const noexcept {
        auto exp = getExponent();
        auto dir = getDirection();
        return 3 * exp + dir;
    }

    // template <std::size_t Count, class UInt>
    // friend constexpr UInt pack_dual_trits(DualTrits const* dual_trits);
    // template <std::size_t Count, class UInt>
    // friend constexpr void unpack_dual_trits(UInt packed, DualTrits* out) noexcept;

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
    /* use a compact representation:
    * a int8_t can have 4 bits.
    * we use 2 bits for exponent and 2 bits for direction.
    * a simple lookup table will do the trick.
    * for direction： 
    * 00 -> 0
    * 01 -> 1
    * 10 -> -1
    * for exponent:
    * 0b0000 -> 0 (3^0)
    * 0b0100 -> 1 (3^1)
    * 0b1000 -> 2 (3^-1
    */
    wide_t storage{}; // 4 bits used
};
    
#endif //PROJECT_FLOAT_DUALTRITS_H
