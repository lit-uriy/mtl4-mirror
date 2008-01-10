/*
 *  matrix_map_view_test_2.cpp
 *  MTL
 *
 *	Test matrix::rscaled_view and matrix::divide_by_view
 *
 *  Created by Hui Li (huil@Princeton.EDU)
 *
 */

#include <iostream>
#include <cmath>
#include <complex>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/matrix/dense2D.hpp>
#include <boost/numeric/mtl/matrix/morton_dense.hpp> 
#include <boost/numeric/mtl/matrix/compressed2D.hpp> 
#include <boost/numeric/mtl/matrix/map_view.hpp>
#include <boost/numeric/mtl/matrix/hermitian_view.hpp>
#include <boost/numeric/mtl/matrix/inserter.hpp>
#include <boost/numeric/mtl/recursion/predefined_masks.hpp>
#include <boost/numeric/mtl/operation/print.hpp>
#include <boost/numeric/mtl/operation/set_to_zero.hpp>
#include <boost/numeric/mtl/operation/conj.hpp>
#include <boost/numeric/mtl/operation/scale.hpp>
#include <boost/numeric/mtl/operation/rscale.hpp>
#include <boost/numeric/mtl/operation/divide_by.hpp>
#include <boost/numeric/mtl/operation/hermitian.hpp>
#include <boost/numeric/mtl/operation/operators.hpp>
#include <boost/numeric/mtl/operation/mult_result.hpp>
#include <boost/numeric/mtl/utility/ashape.hpp>

using namespace mtl;
using namespace std;  

typedef complex<double> ct;

double value(double)
{
    return 7.0;
}

complex<double> value(complex<double>)
{
    return ct(7.0, 1.0);
}

// right scaled value
double rsvalue(double)
{
    return 14.0;
}

ct rsvalue(ct)
{
    return ct(14.0, 2.0);
}

// complex scaled value
ct crsvalue(double)
{
    return ct(0.0, 7.0);
}

ct crsvalue(ct)
{
    return ct(-1.0, 7.0);
}


template <typename Matrix>
void test(Matrix& matrix, const char* name)
{
    set_to_zero(matrix);
    typename Matrix::value_type ref(0);
	
    {
		matrix::inserter<Matrix>  ins(matrix);
		ins(2, 3) << value(ref);
		ins(4, 3) << value(ref) + 1.0;
		ins(2, 5) << value(ref) + 2.0;
    }
	
    cout << "\n\n" << name << "\n";
    cout << "Original matrix:\n" << matrix << "\n";
	
#if 0
    std::cout << "ashape double: " << typeid(typename ashape::ashape<double>::type).name() << "\n";
    std::cout << "ashape matrix: " << typeid(typename ashape::ashape<Matrix>::type).name() << "\n";
    typedef typename ashape::ashape<double>::type   dshape;
    typedef typename ashape::ashape<Matrix>::type   mshape;
	
    std::cout << "ashape mult_op: " << typeid(typename ashape::mult_op<dshape, mshape>::type()).name() << "\n";
    typedef typename ashape::mult_op<dshape, mshape>::type rshape;
	
    
    std::cout << "result type: " << typeid(typename traits::mult_result_aux<double, Matrix, rshape>::type).name() << "\n";
#endif
	
	
	// test rscaled_view
    matrix::rscaled_view<Matrix, double>  rscaled_matrix(matrix, 2.0);
    cout << "matrix  right scaled with 2.0\n" << rscaled_matrix << "\n";
    if (rscaled_matrix(2, 3) != rsvalue(ref)) 
		throw "right scaling wrong";
	
    cout << "matrix  right scaled with 2.0 (as operator)\n" << matrix * 2.0 << "\n";
    if ((matrix * 2.0)(2, 3) != rsvalue(ref)) 
		throw "right scaling wrong";
	
	
    matrix::rscaled_view<Matrix, ct>  crscaled_matrix(matrix, ct(0.0, 1.0));
    cout << "matrix right scaled with i (complex(0, 1))\n" << crscaled_matrix << "\n";
    if (crscaled_matrix(2, 3) != crsvalue(ref)) 
		throw "complex right scaling wrong";
	
    cout << "matrix right scaled with 2.0 (free function)\n" << rscale(matrix, 2.0) << "\n";
    if (rscale(matrix, 2.0)(2, 3) != rsvalue(ref)) 
		throw "scaling wrong";
	
    cout << "matrix right scaled with i (complex(0, 1)) (free function)\n" << rscale(matrix, ct(0.0, 1.0)) << "\n";
    if (rscale(matrix, ct(0.0, 1.0))(2, 3) != crsvalue(ref)) 
		throw "complex right scaling wrong";
	
	// test divide_by_view
    matrix::divide_by_view<Matrix, double>  div_matrix(matrix, 0.5);
    cout << "matrix divide by 0.5\n" << div_matrix << "\n";
    if (div_matrix(2, 3) != rsvalue(ref)) 
		throw "divide_by wrong";
	
    cout << "matrix divide by 0.5 (as operator)\n" << matrix / 0.5 << "\n";
    if ((matrix / 0.5)(2, 3) != rsvalue(ref)) 
		throw "divide_by wrong";
	
	
    matrix::divide_by_view<Matrix, ct>  cdiv_matrix(matrix, ct(0.0, -1.0));
    cout << "matrix divide by -i (complex(0, -1))\n" << cdiv_matrix << "\n";
    if (cdiv_matrix(2, 3) != crsvalue(ref)) 
		throw "complex divide_by wrong";
	
    cout << "matrix divide by 0.5 (free function)\n" << rscale(matrix, 0.5) << "\n";
    if (divide_by(matrix, 0.5)(2, 3) != rsvalue(ref)) 
		throw "scaling wrong";
	
    cout << "matrix divide by -i (complex(0, 1)) (free function)\n" << divide_by(matrix, ct(0.0, -1.0)) << "\n";
    if (divide_by(matrix, ct(0.0, -1.0))(2, 3) != crsvalue(ref)) 
		throw "complex right scaling wrong";
	
}



int test_main(int argc, char* argv[])
{
    unsigned size= 7; 
    if (argc > 1) size= atoi(argv[1]); 
	
    dense2D<double>                                      dr(size, size);
    dense2D<double, matrix::parameters<col_major> >      dc(size, size);
    morton_dense<double, recursion::morton_z_mask>       mzd(size, size);
    morton_dense<double, recursion::doppled_2_row_mask>  d2r(size, size);
    compressed2D<double>                                 cr(size, size);
    compressed2D<double, matrix::parameters<col_major> > cc(size, size);
	
    dense2D<complex<double> >                            drc(size, size);
    compressed2D<complex<double> >                       crc(size, size);
	
	
    test(dr, "Dense row major");
    test(dc, "Dense column major");
    test(mzd, "Morton Z-order");
    test(d2r, "Hybrid 2 row-major");
    test(cr, "Compressed row major");
    test(cc, "Compressed column major");
    test(drc, "Dense row major complex");
    test(crc, "Compressed row major complex");
	
    return 0;
}
