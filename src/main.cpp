#include <iostream>
#include "DualTrits.hpp"

int main() {
    for (uint8_t exponent = 0; exponent < 3; exponent++) {
        for (int direction = 0; direction < 3; direction++) {
            DualTrits digit(exponent, direction);
            std::cout << digit.toFancyString() << std::endl;
            std::cout << std::endl;
            std::cout << "\tAs Packed Bits = "<< digit.asPackedBits().to_string() << std::endl;
            std::cout << "\tRaw Value = "<< digit.asRawPackedBits() << std::endl;
            std::cout << "\tDouble = "<< digit.toDoubleString() << std::endl;
            std::cout << "\tMPreal = "<< digit.toMPrealString() << std::endl;
            std::cout << std::endl;
        }
    }
    return 0;
}
