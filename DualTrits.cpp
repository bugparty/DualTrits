//
// Created by bowman on 2025/10/22.
//

#include "DualTrits.h"
#include <limits>

typedef int8_t wide_t;

std::string DualTrits::toString() const {
    std::bitset<4> bits = this->asBits();
    std::ostringstream oss;
    oss << "exponent = " << bits[3] << bits[2] << ", direction = " << bits[1] << bits[0];
    return oss.str();
}

std::string DualTrits::toFancyString() const {
    std::bitset<4> bits = this->asBits();
    std::ostringstream oss;
    oss << "          ╭────┬────╮\n";
    oss << "DualTrit: │ " << bits[3] << bits[2] << " │ " << bits[1] << bits[0] << " │\n";
    oss << "          ╰────┴────╯\n";
    oss << "            e    d";
    return oss.str();
}

float DualTrits::toFloat() const noexcept {
    return this->to<float>();
}

double DualTrits::toDouble() const noexcept {
    return this->to<double>();
}

mpfr::mpreal DualTrits::toMPreal() const noexcept {
    if (this->direction == 0) {
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
    mpfr::mpreal direction{reinterpt_digit(this->direction)};
    mpfr::mpreal exponent{reinterpt_digit(this->exponent)};
    return direction * mpfr::pow(base,exponent);
}

std::string DualTrits::toFloatString() const {
    return this->toAsString<float>();
}

std::string DualTrits::toDoubleString() const {
    return this->toAsString<double>();
}

std::string DualTrits::toMPrealString() const {
    return toMPreal().toString();
}

DualTrits::compute_t DualTrits::mul3() const {
    //exact compute
    if (isSpecial()) {
        return -1;// not handling special values now 
    }
    if (exponent == 2){ // 3^-1
        return reinterpt_digit(direction);
    }else if (exponent == 0){ // 3^0
        return reinterpt_digit(direction) * 3;
    }else if (exponent == 1){ // 3^1
        return reinterpt_digit(direction) * 9;
    }
    return 0;
}
/*
our number range is -inf, -3,-1,-1/3,0,1/3,1,3,inf
if all multipled by 3, we get -inf,-9,-3,-1,0,1,3,9,inf
if the sum is smaller than -9, return -inf
if the sum is greater than 9, return inf
else, we need to round them back to dual trits

*/
namespace {
    // Sorted array of valid values after multiplying by 3 for rounding purposes
    constexpr int kValidMul3Values[] = {-9, -3, -1, 0, 1, 3, 9};
    constexpr int kValidMul3ValuesSize = 7;
}

DualTrits DualTrits::divide3(DualTrits::compute_t num) const {
    //exact compute
    switch (num){
        case -9:
            return DualTrits(1,2); // -3
        case -3:
            return DualTrits(0,2); // -1
        case -1:
            return DualTrits(2,2); // -1/3
        case 0:
            return DualTrits(0,0); // 0
        case 1:
            return DualTrits(2,1); // 1/3
        case 3:
            return DualTrits(0,1); // 1
        case 9:
            return DualTrits(1,1); // 3
        default:
            return DualTrits(0,0); // should not reach here
    }
}
DualTrits DualTrits::round_mul3(DualTrits::compute_t num) const {
    int l=0, r=kValidMul3ValuesSize;
    while (l < r){
        int mid = l + (r - l) / 2;
        if (kValidMul3Values[mid] < num){
            l = mid + 1;
        }else{
            r = mid;
        }
    }
    //now l is the smallest index s.t. kValidMul3Values[l] >= num
    if (l == 0){
        return DualTrits(2,0); // -inf
    }
    if (l == kValidMul3ValuesSize){
        return DualTrits(1,0); // inf
    }
    if (std::abs(kValidMul3Values[l] - num) < std::abs(kValidMul3Values[l-1] - num)){
        return divide3(kValidMul3Values[l]);
    }else if (std::abs(kValidMul3Values[l] - num) > std::abs(kValidMul3Values[l-1] - num)){
        return divide3(kValidMul3Values[l-1]);
    }else{
        //tie, round to even
        if (kValidMul3Values[l] % 2 == 0){
            return divide3(kValidMul3Values[l]);
        }else{
            return divide3(kValidMul3Values[l-1]);
        }
    }

    return DualTrits(0,0); // should not reach here
}
DualTrits DualTrits::operator+(const DualTrits& other) const {
    //exact compute
    compute_t x,y;
    x = this->mul3();
    y = other.mul3();
    compute_t sum = x + y;
    return round_mul3(sum);
}
DualTrits DualTrits::operator-(const DualTrits& other) const {
    //exact compute
    compute_t x,y;
    x = this->mul3();
    y = other.mul3();
    compute_t diff = x - y;
    return round_mul3(diff);
}
DualTrits DualTrits::operator*(const DualTrits& other) const {
    //exact compute
    return DualTrits{};
}
DualTrits DualTrits::operator/(const DualTrits& other) const {
    //exact compute
    return DualTrits{};
}

void DualTrits::swap(DualTrits& other) noexcept {
    // Swap exponent
    unsigned int temp_exponent = this->exponent;
    this->exponent = other.exponent;
    other.exponent = temp_exponent;
    
    // Swap direction
    unsigned int temp_direction = this->direction;
    this->direction = other.direction;
    other.direction = temp_direction;
}

template<typename T>
[[nodiscard]] constexpr T DualTrits::to() const noexcept {
    if (this->direction == 0) {
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
    wide_t reinterpt_direction = reinterpt_digit(direction);
    T convertedDirection = static_cast<T>(reinterpt_direction);
    T convertedExponent = pow_base<T, BASE>(exponent);
    return convertedDirection * convertedExponent;
}

template<typename T>
[[nodiscard]] std::string DualTrits::toAsString() const {
    std::ostringstream oss;

    if (this->direction == 0) {
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
        //exponent == 2
        if (std::numeric_limits<T>::has_infinity) {
            oss << -std::numeric_limits<T>::infinity();
        } else {
            oss << std::numeric_limits<T>::lowest();
        }
        return oss.str();
    }
    wide_t reinterpt_direction = reinterpt_digit(direction);
    wide_t reinterpt_exponent = reinterpt_digit(exponent);
    T convertedDirection = static_cast<T>(reinterpt_direction);
    T convertedExponent = pow_base<T, BASE>(exponent);

    T result = convertedDirection * convertedExponent;
    oss << "(" << (int) BASE << " ** " << (int) reinterpt_exponent << ") * " << convertedDirection << " = " << result;
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
    return std::bitset<4>(4 * this->exponent + this->direction);
}

unsigned int DualTrits::asRawBits() const noexcept {
    return 4 * this->exponent + this->direction;
}

std::bitset<4> DualTrits::asPackedBits() const noexcept {
    return std::bitset<4>(3 * this->exponent + this->direction);
}

unsigned int DualTrits::asRawPackedBits() const noexcept {
    return 3 * this->exponent + this->direction;
}
