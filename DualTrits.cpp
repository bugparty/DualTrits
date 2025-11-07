//
// Created by bowman on 2025/10/22.
//

#include "DualTrits.h"
#include <limits>

typedef int8_t wide_t;

std::string DualTrits::toString() const noexcept {
    std::bitset<4> bits = this->asBits();
    std::ostringstream oss;
    oss << "exponent = " << bits[3] << bits[2] << ", mantissa = " << bits[1] << bits[0];
    return oss.str();
}

std::string DualTrits::toFancyString() const noexcept {
    std::bitset<4> bits = this->asBits();
    std::ostringstream oss;
    oss << "          ╭────┬────╮\n";
    oss << "DualTrit: │ " << bits[3] << bits[2] << " │ " << bits[1] << bits[0] << " │\n";
    oss << "          ╰────┴────╯\n";
    oss << "            e    m";
    return oss.str();
}

float DualTrits::toFloat() const noexcept {
    return this->to<float>();
}

double DualTrits::toDouble() const noexcept {
    return this->to<double>();
}

mpfr::mpreal DualTrits::toMPreal() const noexcept {
    if (this->mantissa == 0) {
        if (this->exponent == 0) {
            return mpfr::mpreal{0};
        }
        if (this->exponent == 1) {
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

std::string DualTrits::toFloatString() const noexcept {
    return this->toAsString<float>();
}

std::string DualTrits::toDoubleString() const noexcept {
    return this->toAsString<double>();
}

std::string DualTrits::toMPrealString() const noexcept {
    return toMPreal().toString();
}

DualTrits DualTrits::operator+(const DualTrits& other) const {
    //exact compute
    return DualTrits{};
}
DualTrits DualTrits::operator-(const DualTrits& other) const {
    //exact compute
    return DualTrits{};
}
DualTrits DualTrits::operator*(const DualTrits& other) const {
    //exact compute
    return DualTrits{};
}
DualTrits DualTrits::operator/(const DualTrits& other) const {
    //exact compute
    return DualTrits{};
}

template<typename T>
[[nodiscard]] constexpr T DualTrits::to() const noexcept {
    if (this->mantissa == 0) {
        if (this->exponent == 0) {
            return T(0);
        }

        if (this->exponent == 1) {
            if (std::numeric_limits<T>::has_infinity) {
                return std::numeric_limits<T>::infinity();
            }
            return std::numeric_limits<T>::max();
        }
        if (std::numeric_limits<T>::has_infinity) {
            return -std::numeric_limits<T>::infinity();
        }
        return std::numeric_limits<T>::lowest();
    }
    wide_t reinterpt_mantissa = reinterpt_digit(mantissa);
    T convertedMantissa = static_cast<T>(reinterpt_mantissa);
    T convertedExponent = pow_base<T, BASE>(exponent);
    return convertedMantissa * convertedExponent;
}

template<typename T>
[[nodiscard]] std::string DualTrits::toAsString() const noexcept {
    std::ostringstream oss;

    if (this->mantissa == 0) {
        if (this->exponent == 0) {
            oss << T(0);
            return oss.str();
        }
        if (this->exponent == 1) {
            if (std::numeric_limits<T>::has_infinity) {
                oss << std::numeric_limits<T>::infinity();
            } else {
                oss << std::numeric_limits<T>::max();
            }
            return oss.str();
        }

        if (std::numeric_limits<T>::has_infinity) {
            oss << -std::numeric_limits<T>::infinity();
        } else {
            oss << std::numeric_limits<T>::lowest();
        }
        return oss.str();
    }
    wide_t reinterpt_mantissa = reinterpt_digit(mantissa);
    wide_t reinterpt_exponent = reinterpt_digit(exponent);
    T convertedMantissa = static_cast<T>(reinterpt_mantissa);
    T convertedExponent = pow_base<T, BASE>(exponent);

    T result = convertedMantissa * convertedExponent;
    oss << "(" << (int) BASE << " ** " << (int) reinterpt_exponent << ") * " << convertedMantissa << " = " << result;
    return oss.str();
}

template<typename T, wide_t BASE>
[[nodiscard]] T constexpr DualTrits::pow_base(wide_t exp) const noexcept {
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

std::bitset<4> DualTrits::asBits() const noexcept {
    return std::bitset<4>(4 * this->exponent + this->mantissa);
}

std::bitset<4> DualTrits::asPackedBits() const noexcept {
    return std::bitset<4>(3 * this->exponent + this->mantissa);
}

