#include <iostream>
#include "aweights.hpp"

int main() {

    AssignmentWeights<1> w1a(0.0);
    std::cout << "w1a(0.0) => " << w1a.i << ": " << w1a.H[0] << std::endl;
    AssignmentWeights<1> w1b(2.3);
    std::cout << "w1b(2.3) => " << w1b.i << ": " << w1b.H[0] << std::endl;

    AssignmentWeights<2> w2a(0.0);
    std::cout << "w2a(0.0) => " << w2a.i << ": " << w2a.H[0] << " " << w2a.H[1] << std::endl;
    AssignmentWeights<2> w2b(2.3);
    std::cout << "w2a(2.3) => " << w2b.i << ": " << w2b.H[0] << " " << w2b.H[1] << std::endl;

    AssignmentWeights<3> w3a(0.0);
    std::cout << "w3a(0.0) => " << w3a.i << ": " << w3a.H[0] << " " << w3a.H[1] << " " << w3a.H[2] << std::endl;
    AssignmentWeights<3> w3b(2.3);
    std::cout << "w3a(2.3) => " << w3b.i << ": " << w3b.H[0] << " " << w3b.H[1] << " " << w3b.H[2] << std::endl;

    AssignmentWeights<4> w4a(0.0);
    std::cout << "w4a(0.0) => " << w4a.i << ": " << w4a.H[0] << " " << w4a.H[1] << " " << w4a.H[2] << " " << w4a.H[3] << std::endl;
    AssignmentWeights<4> w4b(2.3);
    std::cout << "w4a(2.3) => " << w4b.i << ": " << w4b.H[0] << " " << w4b.H[1] << " " << w4b.H[2] << " " << w4b.H[3] << std::endl;


}