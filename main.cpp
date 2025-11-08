#include <iostream>
#include "DualTrits.h"

int main() {
    std::cout << "================ ALL DualTrits Configurations ================" << std::endl;
    for (unsigned int exponent = 0; exponent < 3; exponent++) {
        for (unsigned int direction = 0; direction < 3; direction++) {
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
    std::cout << "==============================================================" << std::endl;

    std::cout << "================   ALL DualTrits Summations   ================" << std::endl;
    for (unsigned int direction1 = 0; direction1 < 3; direction1++) {
        for (unsigned int exponent1 = 0; exponent1 < 3; exponent1++) {
            for (unsigned int direction2 = 0; direction2 < 3; direction2++) { 
                for (unsigned int exponent2 = 0; exponent2 < 3; exponent2++) {  
                    DualTrits digit1(exponent1, direction1);
                    DualTrits digit2(exponent2, direction2);
                    DualTrits digit3 = digit1 + digit2;
                    
                    auto real = digit1.toMPreal() + digit2.toMPreal();
                    auto calculated = digit3.toMPreal();
                    if (real != calculated) {
                        std::cout << digit1.toFancyString() << std::endl;
                        std::cout << digit2.toFancyString() << std::endl;
                        std::cout << "+ ______________________" << std::endl;
                        std::cout << digit3.toFancyString() << std::endl;
                        std::cout << "\tReal Value = "<< real << std::endl;
                        std::cout << "\tCalculated Value = "<< calculated << std::endl;
                        std::cout << std::endl;
                        std::cout << std::endl;
                    }
                }
            }
        }
    } 
    std::cout << "==============================================================" << std::endl;

    return 0;
}
