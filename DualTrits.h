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

// #ifndef (LITTLE_ENDIAN) || ()
// #define LITTLE_ENDIAN
// #endif

#if defined(LITTLE_ENDIAN) || defined(BIG_ENDIAN)
#else
#define LITTLE_ENDIAN
#endif

class DualTrits {
public:
    typedef uint8_t uwide_t;
    typedef int8_t wide_t;
    static constexpr uwide_t BASE = 3;

    explicit DualTrits() noexcept {
        data.segments.exponent = 0b00;
        data.segments.direction = 0b00;
    }

    explicit DualTrits(const unsigned int e, const unsigned int d) noexcept {
        data.segments.exponent = e;
        data.segments.direction = d;
    }

    explicit DualTrits(const unsigned int raw) noexcept {
        data.raw = raw;
    }

    explicit DualTrits(const DualTrits* dt) noexcept {
        data.raw = dt->data.raw;
    }

    DualTrits(const DualTrits& dt) noexcept {
        data.raw = dt.data.raw;
    }

    DualTrits(DualTrits&& dt) noexcept {
        data.raw = dt.data.raw;
    }

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
    DualTrits operator-() const;
    DualTrits operator*(const DualTrits& other) const;
    DualTrits operator/(const DualTrits& other) const;

    std::bitset<4> asBits() const noexcept;
    unsigned int asRawBits() const noexcept;
    std::bitset<4> asPackedBits() const noexcept;
    unsigned int asRawPackedBits() const noexcept;

    bool isNaN() const noexcept;
    bool isInf() const noexcept;

private:
    template<typename T>
    [[nodiscard]] constexpr T to() const noexcept;

    template<typename T>
    [[nodiscard]] std::string toAsString() const;

    [[nodiscard]] static wide_t constexpr asSignedDigit(uwide_t digit) noexcept {
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

    template<typename T, uwide_t BASE>
    [[nodiscard]] T constexpr pow_base(wide_t exp) const noexcept;

    union {
#ifdef LITTLE_ENDIAN
        struct {
            unsigned int direction : 2;
            unsigned int exponent : 2;
        } segments;
#else
        struct {
            unsigned int exponent : 2;
            unsigned int direction : 2;
        } segments;
#endif

        unsigned int raw : 4;
    } data;
};

#endif //PROJECT_FLOAT_DUALTRITS_H
