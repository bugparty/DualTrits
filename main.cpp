#include <iostream>
#include "DualTrits.h"

int main() {
    for (uint8_t exponent = 0; exponent < 3; exponent++) {
        for (int mantissa = 0; mantissa < 3; mantissa++) {
            DualTrits digit(mantissa, exponent);
            std::cout << "DualTrit = " << digit.toString() << std::endl;
            std::cout << "\tDouble = "<< digit.toDoubleString() << std::endl;
            std::cout << "\tMPreal = "<< digit.toMPrealString() << std::endl;
        }
    }

    return 0;
}
