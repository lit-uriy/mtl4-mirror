#include <iostream>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>

using namespace std;

double f(double) { cout << "double\n"; return 1.0; }
complex<double> f(complex<double>) { cout << "complex\n"; return
complex<double>(1.0, -1.0); }

int test_main(int argc, char* argv[])
{
    using namespace mtl;

    dense_vector<double>                    eig;

    double array[][4]= {{1,  1,   1,  0},
                        {1, -1,  -2,  0},
                        {1, -2,   1,  0},
                        {0,  0,   0, 10}};
    dense2D<double> A(array);
    std::cout << "A=\n" << A << "\n";

    eig= eigenvalue_symmetric(A,22);

    std::cout<<"eigenvalues  ="<< eig <<"\n";

    return 0;
}



