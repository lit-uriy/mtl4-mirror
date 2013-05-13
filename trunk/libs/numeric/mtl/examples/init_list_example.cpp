// Filename: init_list_example.cpp (part of MTL4)

#include <iostream>
#include <boost/numeric/mtl/mtl.hpp>

template <typename Matrix>
void p(const Matrix& A)
{
    std::cout << A;
}

int main()
{
    const mtl::dense2D<double> A={{3, 4}, 
	  	                 {5, 6}};
    mtl::dense2D<double>       B;
    B= {{3, 4}, 
	{5, 6}};
    
    mtl::dense_vector<double> v= {2, 3, 4, 5};

    p(mtl::dense2D<double>{{1, 2}, {2, 3}});

    return 0;
}
