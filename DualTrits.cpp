//
// Created by bowman on 2025/10/22.
//

#include "DualTrits.h"

typedef int8_t wide_t;

std::string DualTrits::toString() {
    std::ostringstream oss;
    oss << "exponent = " << this->exponent << ", mantissa = " << this->mantissa;
    return oss.str();
}

float DualTrits::toFloat() {
    return this->to<float>();
}

double DualTrits::toDouble() {
    return this->to<double>();
}

mpfr::mpreal DualTrits::toMPreal() const {
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

std::string DualTrits::toFloatString() {
    return this->toAsString<float>();
}

std::string DualTrits::toDoubleString() {
    return this->toAsString<double>();
}

std::string DualTrits::toMPrealString() {
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
            return std::numeric_limits<T>::infinity();
        }
        return -std::numeric_limits<T>::infinity();
    }
    auto reinterpt_mantissa = reinterpt_digit(mantissa);
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
            oss << std::numeric_limits<T>::infinity();
            return oss.str();
        }
        oss << -std::numeric_limits<T>::infinity();
        return oss.str();
    }
    auto reinterpt_mantissa = reinterpt_digit(mantissa);
    auto reinterpt_exponent = reinterpt_digit(exponent);
    T convertedMantissa = static_cast<T>(reinterpt_mantissa);
    T convertedExponent = pow_base<T, BASE>(exponent);

    T result = convertedMantissa * convertedExponent;
    oss << "(3 ** " << (int) reinterpt_exponent << ") * " << convertedMantissa << " = " << result;
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
