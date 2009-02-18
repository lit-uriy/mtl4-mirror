#include <iostream>
#include <boost/numeric/mtl/mtl.hpp>

int main(int argc, char* argv[])
{
    using namespace mtl;
    
    typedef std::complex<double>      cdouble;
    const unsigned                    xd= 2, yd= 5, n= xd * yd;
    dense2D<cdouble>                  A(n, n);
    matrix::laplacian_setup(A, xd, yd); 

    // Fill imaginary part of the matrix
    A*= cdouble(1, -1);
    std::cout << "A is\n" << with_format(A, 7, 1) << "\n";

    std::cout << "sub_matrix(A, 2, 4, 1, 7) is\n" 
	      << with_format(sub_matrix(A, 2, 4, 1, 7), 7, 1) << "\n";

    dense2D<cdouble> B= sub_matrix(A, 2, 4, 1, 7);
    B[1][2]= 88;

    std::cout << "A is\n" << with_format(A, 7, 1) << "\n";

    return 0;
}
