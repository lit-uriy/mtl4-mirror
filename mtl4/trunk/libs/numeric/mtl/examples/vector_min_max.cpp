// File: vector_min_max.cpp

#include <iostream>
#include <cmath>
#include <boost/numeric/mtl/mtl.hpp>

int main(int argc, char* argv[])
{
    using mtl::max;

    mtl::dense_vector<double>         v(100);

    for (int i= 0; i < size(v); i++)
	v[i]= double(i+1) * pow(-1.0, i);

    std::cout << "max(v) is " << max(v)<< "\n";
    
    std::cout << "min(v) is " <<  min(v)<< "\n";
    
    std::cout << "max<6>(v) is " <<  max<6>(v)<< "\n";

    return 0;
}

