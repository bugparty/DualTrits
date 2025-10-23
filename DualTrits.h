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
#include <mpreal.h>

class DualTrits {
public:
    typedef int8_t wide_t;
    static constexpr wide_t BASE = 3;

    constexpr explicit DualTrits(wide_t m = 0, int e = 0) noexcept : exponent(e), mantissa(m) {
    }

    std::string toString() {
        std::ostringstream oss;
        oss << "exponent = " << this->exponent << ", mantissa = " << this->mantissa;
        return oss.str();
    }

    float toFloat() {
        return to<float>();
    }

    double toDouble() {
        return to<double>();
    }

    std::string toFloatString() {
        return toAsString<float>();
    }

    std::string toDoubleString() {
        return toAsString<double>();
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

    mpfr::mpreal toMPreal() const {
        if (this->exponent == 0) {
            if (this->mantissa == 0) {
                return mpfr::mpreal{0};
            }
            if (this->mantissa == 1) {
                mpfr::mpreal inf;
                inf.setInf();
                return inf;
            }
            mpfr::mpreal neginf;
            neginf.setInf(-1);
            return mpfr::mpreal{neginf};
        }
        mpfr::mpreal base{BASE};
        mpfr::mpreal mantissa{reinterpt_digit(this->mantissa)};
        mpfr::mpreal exponent{reinterpt_digit(this->exponent)};
        return mantissa * mpfr::pow(base,exponent);
    }

    std::string toMPrealString() {
        return toMPreal().toString();
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
        // std::cout << convertedExponent << " * " << convertedMantissa << " = ";
        return convertedMantissa * convertedExponent;
    }

    template<typename T>
    [[nodiscard]] std::string toAsString() const noexcept {
        std::ostringstream oss;

        if (this->exponent == 0) {
            if (this->mantissa == 0) {
                oss << T(0);
                return oss.str();
            }
            if (this->mantissa == 1) {
                oss << std::numeric_limits<T>::max();
                return oss.str();
            }
            oss << std::numeric_limits<T>::lowest();
            return oss.str();
        }
        auto reinterped_mantissa = reinterpt_digit(mantissa);
        T convertedMantissa = static_cast<T>(reinterped_mantissa);
        T convertedExponent = pow_base<T, BASE>(exponent);

        T result = convertedMantissa * convertedExponent;
        oss << convertedExponent << " * " << convertedMantissa << " = " << result;
        return oss.str();
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


    unsigned int exponent : 2;
    unsigned int mantissa : 2;
};

#endif //PROJECT_FLOAT_DUALTRITS_H
