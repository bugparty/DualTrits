#include <iostream>
#include "DualTrits.h"

int main() {
    for (int e=0;e<3;e++) {
        for (int m=0;m<3;m++) {
            DualTrits digit(m,e);
            std::cout << "e = " << e << ", m = " << m <<
                " DualTrits= "<< digit.toDouble() << std::endl;
        }
    }
    for (int e=0;e<3;e++) {
        for (int m=0;m<3;m++) {
            DualTrits digit(m,e);
            std::cout << "e = " << e << ", m = " << m <<
                " DualTrits MPreal= "<< digit.toMPreal() << std::endl;
        }
    }
    return 0;
}