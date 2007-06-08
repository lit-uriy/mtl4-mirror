// File: compressed2D.cpp

#include <iostream>
#include <boost/numeric/mtl/mtl.hpp>

int main(int argc, char* argv[])
{
    using namespace mtl;

    // CRS matrix
    compressed2D<double>   a(12, 12);

    // Laplace operator discretized on a 3x4 grid
    matrix::laplacian_setup(a, 3, 4);
    std::cout << "a is \n" << a;
    
    // Element access is allowed for reading
    std::cout << "a[3][2] is " << a[3][2] << "\n\n";
    
    // CCS matrix
    compressed2D<float, matrix::parameters<tag::col_major> > b(10, 10);
    // Assign a three times the identity to b
    b= 3;
    std::cout << "b is \n" << b << "\n";

    return 0;
}

