//
// Created by bowman on 2025/10/22.
//

#include "DualTrits.h"
#include <cstdlib>
#include <limits>

typedef uint8_t uwide_t;
typedef int8_t wide_t;

// Raw binary form of DualTrits
const unsigned int RAW_INF           = 0b0100u;
const unsigned int RAW_THREE         = 0b0101u;
const unsigned int RAW_ONE           = 0b0001u;
const unsigned int RAW_ONE_THIRD     = 0b1001u;
const unsigned int RAW_ZERO          = 0b0000u;
const unsigned int RAW_NEG_ONE_THIRD = 0b1010u;
const unsigned int RAW_NEG_ONE       = 0b0010u;
const unsigned int RAW_NEG_THREE     = 0b0110u;
const unsigned int RAW_NEG_INF       = 0b1000u;
const unsigned int RAW_NAN           = 0b1111u;

// Raw binary form of the direction trit
const unsigned int D_POS  = 0b01u;
const unsigned int D_ZERO = 0b00u;
const unsigned int D_NEG  = 0b10u;
const unsigned int D_NAN  = 0b11u;

// Raw binary form of the exponent trit
const unsigned int E_ONE_THIRD = 0b10u;
const unsigned int E_ONE       = 0b00u;
const unsigned int E_THREE     = 0b01u;
const unsigned int E_NAN       = 0b11u;

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
    oss << "DualTrit: │ " << bits[3] << bits[2] << " │ " << bits[1] << bits[0] << " │" << " " << this->toDoubleString() << "\n";
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
    // NaN
    if (this->data.raw == RAW_NAN) {
        return mpfr::mpreal("nan");
    }

    if (this->data.segments.direction == D_ZERO) {
        if (this->data.segments.exponent == E_ONE) {
            return mpfr::mpreal{0};
        }
        if (this->data.segments.exponent == E_THREE) {
            mpfr::mpreal inf;
            inf.setInf();
            return inf;
        }
        mpfr::mpreal neginf;
        neginf.setInf(-1);
        return mpfr::mpreal{neginf};
    }
    mpfr::mpreal base{BASE};
    mpfr::mpreal direction{asSignedDigit(this->data.segments.direction)};
    mpfr::mpreal exponent{asSignedDigit(this->data.segments.exponent)};
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

DualTrits DualTrits::operator+(const DualTrits& other) const {
    //exact compute

    // Identity Properties
    if (this->data.raw == RAW_ZERO) {
        return DualTrits(other);
    }
    if (other.data.raw == RAW_ZERO) {
        return DualTrits(this);
    }

    // Infinity Operations
    if (this->data.raw == RAW_NEG_INF) {
        // Indeterminate form
        if (other.data.raw == RAW_INF) {
            return DualTrits(RAW_NAN);
        }
        return DualTrits(RAW_NEG_INF);
    }
    if (other.data.raw == RAW_NEG_INF) {
        // Indeterminate form
        if (this->data.raw == RAW_INF) {
            return DualTrits(RAW_NAN);
        }
        return DualTrits(RAW_NEG_INF);
    }
    if (this->data.raw == RAW_INF) {
        // Indeterminate form
        if (other.data.raw == RAW_NEG_INF) {
            return DualTrits(RAW_NAN);
        }
        return DualTrits(RAW_INF);
    }
    if (other.data.raw == RAW_INF) {
        // Indeterminate form
        if (this->data.raw == RAW_NEG_INF) {
            return DualTrits(RAW_NAN);
        }
        return DualTrits(RAW_INF);
    }

    // -3 + other operations
    if (this->data.raw == RAW_NEG_THREE) {
        // Underflow
        // -3 + (-1) == -3 + (-1/3) = -inf
        // => -3 + neg = -inf
        if (other.data.segments.direction == D_NEG) {
            return DualTrits(RAW_NEG_INF);
        }
        // Cancel out
        // -3 + 3
        if (other.data.raw == RAW_THREE) {
            return DualTrits(RAW_ZERO);
        }

        // Small increments round to -1
        // -3 + 1/3 = -1
        return DualTrits(RAW_NEG_ONE);
    }

    // 3 + other operations
    if (this->data.raw == RAW_THREE) {
        // Overflow
        // 3 + 1 == 3 + 1/2 == inf
        // => 3 + pos == inf
        if (other.data.segments.direction == D_POS) {
            return DualTrits(RAW_INF);
        }

        // Cancel out
        // 3 - 3 == 0
        if (other.data.raw == RAW_NEG_THREE) {
            return DualTrits(RAW_ZERO);
        }

        // Small decrements round to 1
        // 3 + -1 == 3 + -1/3 == 1
        // => 3 + small decrement == 1
        return DualTrits(RAW_ONE);
    }
    
    // -1/3 + other operations
    if (this->data.raw == RAW_NEG_ONE_THIRD) {
        // Overflow
        // -1/3 + (-3) == -inf
        if (other.data.raw == RAW_NEG_THREE) {
            return DualTrits(RAW_NEG_INF);
        }

        // Cancel out
        // -1/3 + 1/3 == 0
        if (other.data.raw == RAW_ONE_THIRD) {
            return DualTrits(RAW_ZERO);
        }

        // Small decrements round to -1
        // -1/3 + (-1/3) == -1/3 + (-1) == -1
        // => -1/3 + small decrement == -1
        if (other.data.raw == RAW_NEG_ONE_THIRD || other.data.raw == RAW_NEG_ONE) {
            return DualTrits(RAW_NEG_ONE);
        } 
        
        // Medium increments round to 1/3
        // -1/3 + 1 == 1/3
        if (other.data.raw == RAW_ONE) {
            return DualTrits(RAW_ONE_THIRD);
        }

        // Large increments round to 3
        // -1/3 + 3 == 3
        return DualTrits(RAW_THREE);
    }

    // 1/3 + other operations
    if (this->data.raw == RAW_ONE_THIRD) {
        // Overflow
        // 1/3 + 3 == inf
        if (other.data.raw == RAW_THREE) {
            return DualTrits(RAW_INF);
        }

        // Cancel out
        // 1/3 + (-1/3)
        if (other.data.raw == RAW_NEG_ONE_THIRD) {
            return DualTrits(RAW_ZERO);
        }

        // Small increments round to 1
        // 1/3 + 1/3 == 1/3 + 1 == 1
        if (other.data.raw == RAW_ONE_THIRD || other.data.raw == RAW_ONE) {
            return DualTrits(RAW_ONE);
        }
        
        // Medium decrements round to -1/3
        // 1/3 + (-1) == -1/3
        if (other.data.raw == RAW_NEG_ONE) {
            return DualTrits(RAW_NEG_ONE_THIRD);
        }

        // Large decrements round to -3
        // 1/3 + (-3) == -3
        return DualTrits(RAW_NEG_THREE);
    } 

    // -1 + other operations
    if (this->data.raw == RAW_NEG_ONE) {
        // Underflow
        // -1 + (-3) == -inf
        if (other.data.raw == RAW_NEG_THREE) {
            return DualTrits(RAW_NEG_INF);
        }

        // Cancel out
        // -1 + 1 == 0
        if (other.data.raw == RAW_ONE) {
            return DualTrits(RAW_ZERO);
        }

        // Small decrements round to -3
        // -1 + (-1) == -1 + (-1/3) == 3
        if (other.data.raw == RAW_NEG_ONE || other.data.raw == RAW_NEG_ONE_THIRD) {
            return DualTrits(RAW_NEG_THREE);
        }

        // Small increments round to -1/3
        // -1 + 1/3 == -1/3
        if (other.data.raw == RAW_ONE_THIRD) {
            return DualTrits(RAW_NEG_ONE_THIRD);
        }
        
        // Large increments round to 1
        // -1 + 3 == 1
        return DualTrits(RAW_ONE);
    }

    // 1 + other operations
    if (this->data.raw == RAW_ONE) {
        // Overflow
        // 1 + 3 == inf
        if (other.data.raw == RAW_THREE) {
            return DualTrits(RAW_INF);
        }

        // Cancel out
        // 1 + (-1) == 0
        if (other.data.raw == RAW_NEG_ONE) {
            return DualTrits(RAW_ZERO);
        }

        // Small increments round to 3
        // 1 + 1 == 1 + 1/3 == 3
        if (other.data.raw == RAW_ONE || other.data.raw == RAW_ONE_THIRD) {
            return DualTrits(RAW_THREE);
        }

        // Small decrements round to 1/3
        // 1 + (-1/3) = 1/3
        if (other.data.raw == RAW_NEG_ONE_THIRD) {
            return DualTrits(RAW_ONE_THIRD);
        }
        
        // Large decrements round to -1
        // 1 + (-3) == -1
        return DualTrits(RAW_NEG_ONE);
    }

    // In the event of any unsupported operation,
    // return nan.
    return DualTrits(RAW_NAN);
}

DualTrits DualTrits::operator-() const {
    //exact compute

    DualTrits result(this);

    // Identiy Property
    if (this->data.raw == 0) {
        return result;
    }

    // +/- inf
    if (this->data.segments.direction == D_ZERO) {
        result.data.segments.exponent = ~result.data.segments.exponent;
        return result;
    }

    // +/- value
    result.data.segments.direction = ~result.data.segments.direction;
    return result;
}

DualTrits DualTrits::operator-(const DualTrits& other) const {
    //exact compute

    // Same as addition, but negate second term
    return *this + -other;
}

DualTrits DualTrits::operator*(const DualTrits& other) const {
    //exact compute
    return DualTrits();
}

DualTrits DualTrits::operator/(const DualTrits& other) const {
    //exact compute
    return DualTrits();
}

template<typename T>
[[nodiscard]] constexpr T DualTrits::to() const noexcept {
    // Convert to NaN
    if (this->data.raw == RAW_NAN) {
        if (std::numeric_limits<T>::has_quiet_NaN) {
            return std::numeric_limits<T>::quiet_NaN();
        }
        return 0;
    }

    // DualTrit == 0bXX00 means we have -inf, 0, or inf
    if (this->data.segments.direction == D_ZERO) {

        // Convert to 0
        // DualTrit == 0b0000
        if (this->data.segments.exponent == E_ONE) {
            return T(0);
        }

        // Convert to inf
        // DualTrit == 0b0100
        if (this->data.segments.exponent == E_THREE) {
            if (std::numeric_limits<T>::has_infinity) {
                return std::numeric_limits<T>::infinity();
            }
            return std::numeric_limits<T>::max();
        }

        // Convert to -inf
        // DualTrit == 0b1000
        if (std::numeric_limits<T>::has_infinity) {
            return -std::numeric_limits<T>::infinity();
        }
        return std::numeric_limits<T>::lowest();
    }

    // Convert unsigned trits from 0, 1, 2
    // to signed ternary values   -1, 0, 1
    T signedDirection = static_cast<T>(asSignedDigit(this->data.segments.direction));
    T signedExponent = static_cast<T>(asSignedDigit(this->data.segments.exponent));

    // Return (BASE ** signed_exponent) * signed_direction
    return pow_base<T, BASE>(signedExponent) * signedDirection;
}

template<typename T>
[[nodiscard]] std::string DualTrits::toAsString() const {
    std::ostringstream oss;

    // Convert to NaN
    if (this->data.raw == RAW_NAN) {
        oss << "nan";
        return oss.str();
    }

    // DualTrit == 0bXX00 means we have -inf, 0, or inf
    if (this->data.segments.direction == D_ZERO) {

        // Convert to 0
        // DualTrit == 0b0000
        if (this->data.segments.exponent == E_ONE) {
            oss << T(0);
            return oss.str();
        }

        // Convert to inf
        // DualTrit == 0b0100
        if (this->data.segments.exponent == E_THREE) {
            if (std::numeric_limits<T>::has_infinity) {
                oss << std::numeric_limits<T>::infinity();
            } else {
                oss << std::numeric_limits<T>::max();
            }
            return oss.str();
        }

        // Convert to -inf
        // DualTrit == 0b1000
        if (std::numeric_limits<T>::has_infinity) {
            oss << -std::numeric_limits<T>::infinity();
        } else {
            oss << std::numeric_limits<T>::lowest();
        }
        return oss.str();
    }

    // Convert unsigned trits from 0, 1, 2
    // to signed ternary values   -1, 0, 1
    T signedDirection = static_cast<T>(asSignedDigit(this->data.segments.direction));
    T signedExponent = static_cast<T>(asSignedDigit(this->data.segments.exponent));

    // Return (BASE ** signed_exponent) * signed_direction
    T result = pow_base<T, BASE>(signedExponent) * signedDirection;
    oss << "(" << (unsigned int) BASE << " ** " << signedExponent << ") * " << signedDirection << " = " << result;
    return oss.str();
}

template<typename T, uwide_t BASE>
[[nodiscard]] T constexpr DualTrits::pow_base(wide_t exp) const noexcept {
    switch (exp) {
        // No need to call std::pow, will always be equal to 1
        case 0:
            return 1;

        // Call std::pow, but at compile time
        case -1:
        case 1:
            return std::pow(BASE, exp);

        // Invalid exponent
        default:
            return 0;
    }
}

std::bitset<4> DualTrits::asBits() const noexcept {
    return std::bitset<4>(4 * this->data.segments.exponent + this->data.segments.direction);
}

unsigned int DualTrits::asRawBits() const noexcept {
    return 4 * this->data.segments.exponent + this->data.segments.direction;
}

std::bitset<4> DualTrits::asPackedBits() const noexcept {
    return std::bitset<4>(3 * this->data.segments.exponent + this->data.segments.direction);
}

unsigned int DualTrits::asRawPackedBits() const noexcept {
    return 3 * this->data.segments.exponent + this->data.segments.direction;
}

bool DualTrits::isNaN() const noexcept {
    return this->data.raw == RAW_NAN;
}

bool DualTrits::isInf() const noexcept {
    return this->data.raw == RAW_INF || this->data.raw == RAW_NEG_INF;
}
