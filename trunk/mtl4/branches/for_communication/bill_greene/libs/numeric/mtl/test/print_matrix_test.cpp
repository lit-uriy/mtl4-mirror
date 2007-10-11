// $COPYRIGHT$

#include <iostream>
#include <cmath>
#include <boost/test/minimal.hpp>

#include <boost/numeric/mtl/matrix/dense2D.hpp>
#include <boost/numeric/mtl/matrix/morton_dense.hpp> 
#include <boost/numeric/mtl/matrix/compressed2D.hpp> 
#include <boost/numeric/mtl/matrix/transposed_view.hpp>

#include <boost/numeric/mtl/recursion/bit_masking.hpp>
#include <boost/numeric/mtl/recursion/predefined_masks.hpp>
#include <boost/numeric/mtl/operation/print.hpp>


using namespace mtl;
using namespace mtl::recursion;
using namespace std;  

template <typename Matrix>
void print_matrix(Matrix& matrix)
{ 
    using std::cout;
    for (int i=0 ; i<matrix.num_rows(); i++ ){
	for(int j=0; j<matrix.num_cols();  j++ ){
	    cout.fill (' '); cout.width (8); cout.precision (5); cout.flags (ios_base::left);
	    cout << showpoint <<  matrix[i][j] <<"  ";
	}
	cout << endl;
    }
}


template <typename Matrix>
void test(Matrix& matrix, const char* name)
{
    {
	matrix::inserter<Matrix> ins(matrix);
	for (int i= 0; i < matrix.num_rows(); i++)
	    for (int j= 0; j < matrix.num_cols(); j++)
		if ((i + j) & 1)
		    ins(i, j) << i + 2*j;
    }

    std::cout << "\n" << name << "\n";
    mtl::print_matrix(matrix);

    transposed_view<Matrix> trans(matrix);
    std::cout << "Transposed" << "\n";
    mtl::print_matrix(trans);

    std::cout << "with <<" << "\n"
	      << trans << "\n";

    std::cout << "with << and formatted" << "\n"
	      << mtl::with_format(trans, 7, 4) << "\n";
}


int test_main(int argc, char* argv[])
{
    dense2D<double>                                dr(5, 7);
    dense2D<double, matrix::parameters<col_major> > dc(5, 7);
    morton_dense<double,  morton_mask>             md(5, 7);
    morton_dense<double,  doppled_16_row_mask>     d16r(5, 7);
    compressed2D<double>                           comp(5, 7);

    test(dr, "Dense row major");
    test(dc, "Dense column major");
    test(md, "Morton N-order");
    test(d16r, "Hybrid 16 row-major");
    test(comp, "compressed2D");

    return 0;
}





