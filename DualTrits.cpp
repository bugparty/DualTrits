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
if all multiplied by 3, we get -inf,-9,-3,-1,0,1,3,9,inf
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
    int l = 0, r = kValidMul3ValuesSize;
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (kValidMul3Values[mid] < num) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    //now l is the smallest index s.t. kValidMul3Values[l] >= num
    if (l == 0) {
        return DualTrits(2,0); // -inf
    }
    if (l == kValidMul3ValuesSize) {
        return DualTrits(1,0); // inf
    }
    if (std::abs(kValidMul3Values[l] - num) < std::abs(kValidMul3Values[l-1] - num)){
        return divide3(kValidMul3Values[l]);
    } else if (std::abs(kValidMul3Values[l] - num) > std::abs(kValidMul3Values[l-1] - num)){
        return divide3(kValidMul3Values[l-1]);
    }else{
        //tie, round half away from zero (choose the one with larger absolute value)
        // because if we do 1+1, if we do tie to even, it still ends up to 1, which doesn't make sense

        if (std::abs(kValidMul3Values[l]) > std::abs(kValidMul3Values[l-1])){
            return divide3(kValidMul3Values[l]);
        } else {
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

/*
 * Multiplication operator for DualTrits
 * 
 * Mathematical formula:
 *   a = direction_a * 3^exponent_a
 *   b = direction_b * 3^exponent_b
 *   a * b = (direction_a * direction_b) * 3^(exponent_a + exponent_b)
 * 
 * Where exponent encoding: 0 -> 0, 1 -> 1, 2 -> -1
 *        direction encoding: 0 -> 0, 1 -> 1, 2 -> -1
 * 
 * Algorithm:
 * 1. Handle special cases (0, inf, -inf)
 * 2. Compute new direction (sign) as product of directions
 * 3. Compute new exponent as sum of exponents
 * 4. Handle overflow/underflow of exponent range
 * 5. Construct result DualTrits
 */
DualTrits DualTrits::operator*(const DualTrits& other) const {
    // Handle multiplication with zero (only if it's truly zero: exp=0, dir=0)
    if ((direction == 0 && exponent == 0) || (other.direction == 0 && other.exponent == 0)) {
        return DualTrits(0, 0); // 0
    }
    
    // Handle special values (inf, -inf)
    if (isSpecial() || other.isSpecial()) {
        // inf * positive = inf, inf * negative = -inf
        // Determine sign of result
        // For special values: exponent=1 means +inf, exponent=2 means -inf
        int sign_a = isSpecial() ? (exponent == 1 ? 1 : -1) : reinterpt_digit(direction);
        int sign_b = other.isSpecial() ? (other.exponent == 1 ? 1 : -1) : reinterpt_digit(other.direction);
        int result_sign = sign_a * sign_b;
        
        // Result is always infinity with appropriate sign
        if (result_sign > 0) {
            return DualTrits(1, 0); // +inf
        } else {
            return DualTrits(2, 0); // -inf
        }
    }
    
    // Normal case: both are finite non-zero values
    // Compute new direction (sign)
    int dir_a = reinterpt_digit(direction);
    int dir_b = reinterpt_digit(other.direction);
    int new_dir_val = dir_a * dir_b; // -1, 0, or 1
    
    // Compute new exponent
    int exp_a = reinterpt_exponent(exponent);
    int exp_b = reinterpt_exponent(other.exponent);
    int new_exp_val = exp_a + exp_b;
    
    // Handle exponent overflow (> 1 means result >= 3^2 = 9)
    if (new_exp_val > 1) {
        // Overflow to infinity
        // Positive overflow -> +inf (1,0), Negative overflow -> -inf (2,0)
        if (new_dir_val > 0) {
            return DualTrits(1, 0); // +inf
        } else {
            return DualTrits(2, 0); // -inf
        }
    }
    
    // Handle exponent underflow (< -1 means result <= 3^-2 = 1/9)
    if (new_exp_val < -1) {
        // 1/9 is not representable, round to 0 or 1/3
        // Since 1/9 ≈ 0.111, it's closer to 0 than to 1/3 ≈ 0.333
        // But with round-away-from-zero for ties, and 1/9 to 0 distance = 1/9
        // 1/9 to 1/3 distance = 2/9, so round to 0
        return DualTrits(0, 0); // 0
    }
    
    // Construct result within valid range
    wide_t new_exp = encode_exponent(new_exp_val);
    wide_t new_dir = encode_direction(new_dir_val);
    
    return DualTrits(new_exp, new_dir);
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

// Helper function: Convert exponent encoding to integer value
// 0 -> 0, 1 -> 1, 2 -> -1
int DualTrits::reinterpt_exponent(wide_t exp) noexcept {
    if (exp == 0) return 0;   // 3^0
    if (exp == 1) return 1;   // 3^1
    if (exp == 2) return -1;  // 3^-1
    return 0;
}

// Helper function: Convert integer exponent value to encoding
// 0 -> 0, 1 -> 1, -1 -> 2
DualTrits::wide_t DualTrits::encode_exponent(int exp_val) noexcept {
    if (exp_val == 0) return 0;
    if (exp_val == 1) return 1;
    if (exp_val == -1) return 2;
    return 0; // Should not happen
}

// Helper function: Convert integer direction value to encoding
// 0 -> 0, 1 -> 1, -1 -> 2
DualTrits::wide_t DualTrits::encode_direction(int dir_val) noexcept {
    if (dir_val == 0) return 0;
    if (dir_val > 0) return 1;  // positive
    return 2;  // negative
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
