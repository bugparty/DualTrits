#include <iostream>
#include "DualTrits.h"

int main() {
    for (uint8_t exponent = 0; exponent < 3; exponent++) {
        for (int mantissa = 0; mantissa < 3; mantissa++) {
            DualTrits digit(mantissa, exponent);
            std::cout << digit.toFancyString() << std::endl;
            std::cout << std::endl;
            std::cout << "\tAs Packed Bits = "<< digit.asPackedBits().to_string() << std::endl;
            std::cout << "\tDouble = "<< digit.toDoubleString() << std::endl;
            std::cout << "\tMPreal = "<< digit.toMPrealString() << std::endl;
            std::cout << std::endl;
        }
    }
    return 0;
}
