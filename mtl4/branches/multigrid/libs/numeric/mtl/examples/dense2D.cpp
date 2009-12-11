// File: dense2D.cpp

#include <iostream>
#include <boost/numeric/mtl/mtl.hpp>

int main(int argc, char* argv[])
{
    using namespace mtl;

    // a is a row-major matrix
    dense2D<double>               a(10, 10);

    // Matrices are not initialized by default
    set_to_zero(a);

    // Assign a value to a matrix element
    a(2, 3)= 7.0;

    // You can also use a more C-like notation
    a[2][4]= 3.0;

    std::cout << "a is \n" << a << "\n";
    
    // b is a column-major matrix
    dense2D<float, matrix::parameters<tag::col_major> > b(10, 10);

    // Assign a three times the identity to b
    b= 3;
    std::cout << "b is \n" << b << "\n";

    return 0;
}

