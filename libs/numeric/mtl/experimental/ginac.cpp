#include "boost/numeric/mtl/mtl.hpp"
#include "ginac/ginac.h"
#include <iostream>

using namespace mtl;
using namespace GiNaC;

//libs:-lcln -lginac
int main(int argc, char* argv[])
{
	dense2D< ex > exMat(2,2);
	symbol r("r");
	exMat(0,0)=1;exMat(0,1)=r;
	exMat(1,0)=0; exMat(1,1)=3;

	std::cout << "mat:\n" << exMat;

	dense_vector< ex > exVec(2);

	return 0;
}
