#include <iostream>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>

int test_main(int argc, char* argv[])
{
    using namespace mtl;
    dense2D<double>       A(5, 5), Hess(5, 5);  A= 0.0;  
    
    A[0][1] = 3;  A[1][4] = 7; A[0][0] = 1; A[4][4] = 17;
    A[2][3] = -2; A[2][4] = 5; A[4][0] = 2; A[4][1] = 3;
    A[3][2] = 4;
    
    Hess= hessenberg(A);
    std::cout<< "Hessenberg=\n" << Hess << "\n";
    Hess= extract_householder_hessenberg(A);
    std::cout<< "extract_householder_hessenberg=\n" << Hess << "\n";
    Hess= extract_hessenberg(A);
    std::cout<< "extract_hessenberg=\n" << Hess << "\n";
    Hess= householder_hessenberg(A);
    std::cout<< "householder_hessenberg=\n" << Hess << "\n";
    Hess= hessenberg_factors(A);
    std::cout<< "hessenberg_factors=\n" << Hess << "\n";
    Hess= hessenberg_q(A);
    std::cout<< "hessenberg_q=\n" << Hess << "\n";

   return 0;
}