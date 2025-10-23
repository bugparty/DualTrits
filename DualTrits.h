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
#include "mpreal.h"
using mpfr::mpreal;
class DualTrits {
public:
    using wide_t =  int8_t;
    static constexpr wide_t BASE = 3;
    constexpr explicit DualTrits(wide_t m = 0, int e = 0) noexcept : exponent(e), mantissa(m) {
    }

    float toFloat() {
        return to<float>();
    }

    double toDouble() {
        return to<double>();
    }
    DualTrits operator+(const DualTrits& other) const {
        //exact compute
        return DualTrits{};
    }
    DualTrits operator-(const DualTrits& other) const {
        //exact compute
        return DualTrits{};
    }
    DualTrits operator*(const DualTrits& other) const {
        //exact compute
        return DualTrits{};
    }
    DualTrits operator/(const DualTrits& other) const {
        //exact compute
        return DualTrits{};
    }
    mpreal toMPreal() const {
        if (this->exponent == 0) {
            if (this->mantissa == 0) {
                return mpreal{0};
            }
            if (this->mantissa == 1) {
                mpreal inf;
                inf.setInf();
                return inf;
            }
            mpreal neginf;
            neginf.setInf(-1);
            return mpreal{neginf};
        }
        mpreal base{BASE};
        mpreal mantissa{reinterpt_digit(this->mantissa)};
        mpreal exponent{reinterpt_digit(this->exponent)};
        return mantissa * mpfr::pow(base,exponent);
    }


private:
    template<typename T>
    [[nodiscard]] constexpr T to() const noexcept {
        if (this->exponent == 0) {
            if (this->mantissa == 0) {
                return T(0);
            }
            if (this->mantissa == 1) {
                return std::numeric_limits<T>::max();
            }
            return std::numeric_limits<T>::lowest();
        }
        auto reinterped_mantissa = reinterpt_digit(mantissa);
        T convertedMantissa = static_cast<T>(reinterped_mantissa);
        T convertedExponent = pow_base<T, BASE>(exponent);
        std::cout << convertedExponent << " * " << convertedMantissa << " = ";
        return convertedMantissa * convertedExponent;
    }

    [[nodiscard]] static wide_t constexpr reinterpt_digit(wide_t digit) noexcept {
        if (digit == 2) {
            return -1;
        }
        if (digit == 1) { return 1; }
        if (digit == 0) { return 0;}
        return 0;
    }

    template<typename T, wide_t BASE>
    [[nodiscard]] T constexpr pow_base(wide_t exp) const noexcept {
        switch (exp) {
            case 0:
                return 1;
            case 1:
                return std::pow(BASE, 1);
            case 2:
                return std::pow(BASE, -1);
            default:
                return 0;
        }
    }


    __uint128_t exponent;
    __uint128_t mantissa;
};


#endif //PROJECT_FLOAT_DUALTRITS_H
