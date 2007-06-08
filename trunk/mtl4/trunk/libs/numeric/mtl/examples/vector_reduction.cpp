// File: vector_reduction.cpp

#include <iostream>
#include <boost/numeric/mtl/mtl.hpp>

int main(int argc, char* argv[])
{
    using namespace mtl;

    typedef std::complex<double>  cdouble;
    dense_vector<cdouble>         v(10000);

    for (int i= 0; i < size(v); i++)
	v[i]= cdouble(i+1, 10000-i);

    std::cout << "sum(v) is " << sum(v)<< "\n";
    
    std::cout << "product(v) is " <<  product(v)<< "\n";
    
    std::cout << "sum<6>(v) is " <<  sum<6>(v)<< "\n";

    return 0;
}

