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

template <typename Matrix, typename Sub>
void substitute(Matrix& A, Sub sub)
{
    for (int i= 0; i < 4; i++)
	for (int j= 0; j < 5; j++)
	    A[i][j]= A[i][j].subs(sub);
}


//libs:-lcln -lginac
// g++ ginac.cpp -lcln -lginac  -o ginac -I$MTL
int main(int argc, char* argv[])
{
    dense2D< ex > A(4, 5);

    symbol r("r"), s("s");

    A= 1, 0, 2, -3, 2,
	-2, 1, 0, 2, -1,
	-1, 2*r, 6, -5, 4,
	1, 1, 6, r, s;

    std::cout << "A:\n" << A;

    eliminate(A, 0, 0);
    std::cout << "A:\n" << A;

    eliminate(A, 1, 2);
    std::cout << "A:\n" << A;

    swap_column(A, 1, 2);
    std::cout << "A:\n" << A;

    substitute(A, lst(r == 2, s == 6));
    std::cout << "A:\n" << A;


#if 0


	dense_vector< ex > exVec(2);
#endif 
	return 0;
}
