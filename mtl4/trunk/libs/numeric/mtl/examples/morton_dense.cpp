// File: morton_dense.cpp

#include <iostream>
#include <boost/numeric/mtl/mtl.hpp>

int main(int argc, char* argv[])
{
    using namespace mtl;

    // Z-order matrix
    morton_dense<double, recursion::morton_z_mask>  a(10, 10);

    set_to_zero(a);
    a(2, 3)= 7.0;
    a[2][4]= 3.0;
    std::cout << "a is \n" << a << "\n";
    
    // b is a N-order matrix with column-major 4x4 blocks
    morton_dense<float, recursion::doppler_4_col_mask> b(10, 10);

    // Assign a three times the identity to b
    b= 3;
    std::cout << "b is \n" << b << "\n";

    return 0;
}

