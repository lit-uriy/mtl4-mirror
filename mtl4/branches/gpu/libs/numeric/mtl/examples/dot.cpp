// File: dot.cpp

#include <iostream>
#include <boost/numeric/mtl/mtl.hpp>

int main(int argc, char* argv[])
{
    using namespace mtl;

    typedef std::complex<double>  cdouble;
    dense_vector<cdouble>         v(10000), x(10, cdouble(3, 2));
    dense_vector<double>          w(10000);

    for (unsigned i= 0; i < size(v); i++)
	v[i]= cdouble(i+1, 10000-i), w[i]= 2 * i + 2;

    std::cout << "dot(v, w) is " << dot(v, w)<< "\n";
    
    std::cout << "dot<6>(v, w) is " <<  dot<6>(v, w)<< "\n";
    
    std::cout << "conj(x) is " <<  conj(x)<< "\n";

    return 0;
}

