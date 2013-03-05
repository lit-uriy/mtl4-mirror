#include "boost/numeric/mtl/mtl.hpp"
#include "ginac/ginac.h"
#include <iostream>

using namespace mtl;
using namespace GiNaC;

using mtl::iall;

template <typename Matrix>
void eliminate(Matrix& A, int r, int c)
{
    ex pivot(A[r][c]);
    for (int i= r + 1; i < 4; i++) {
	ex factor(A[i][c] / pivot);
	A[i][iall]-= factor * A[r][iall];
    }
}

template <typename Matrix>
void swap_column(Matrix& A, int c1, int c2)
{
    dense_vector<ex> tmp(clone(A[iall][c1]));
    A[iall][c1]= A[iall][c2];
    A[iall][c2]= tmp;
}

//libs:-lcln -lginac
// g++ ginac.cpp -lcln -lginac  -o ginac -I$MTL
int main(int argc, char* argv[])
{
    dense2D< ex > A(4, 4);

    symbol r("r"), s("s");

    A= 1, 0, 2, -3,
	-2, 1, 0, 2,
	-1, 2*r, 6, -5,
	1, 1, 6, r;

    std::cout << "A:\n" << A;

    eliminate(A, 0, 0);
    std::cout << "A:\n" << A;

    eliminate(A, 1, 2);
    std::cout << "A:\n" << A;

    swap_column(A, 1, 2);
    std::cout << "A:\n" << A;

#if 0


	dense_vector< ex > exVec(2);
#endif 
	return 0;
}
