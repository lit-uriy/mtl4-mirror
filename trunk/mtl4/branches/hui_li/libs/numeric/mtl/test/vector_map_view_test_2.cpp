/*
 *  vector_map_view_test_2.cpp
 *  MTL
 *
 *	Test vector::rscaled_view and vector::divide_by_view
 *
 *  Created by Hui Li (huil@Princeton.EDU)
 *
 */


#include <iostream>
#include <cmath>
#include <complex>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/vector/dense_vector.hpp>
#include <boost/numeric/mtl/vector/map_view.hpp>
#include <boost/numeric/mtl/operation/print.hpp>
#include <boost/numeric/mtl/operation/set_to_zero.hpp>
#include <boost/numeric/mtl/operation/conj.hpp>
#include <boost/numeric/mtl/operation/rscale.hpp>
#include <boost/numeric/mtl/operation/divide_by.hpp>

#if 0
#include <boost/numeric/mtl/operation/hermitian.hpp>
#endif

using namespace mtl;
using std::cout;  using std::complex;

typedef complex<double> ct;

double value(double)
{
    return 7.0;
}

complex<double> value(complex<double>)
{
    return ct(7.0, 1.0);
}

// rscaled value
double rsvalue(double)
{
    return 14.0;
}

ct rsvalue(ct)
{
    return ct(14.0, 2.0);
}

// complex rscaled value
ct crsvalue(double)
{
    return ct(0.0, 7.0);
}

ct crsvalue(ct)
{
    return ct(-1.0, 7.0);
}


template <typename Vector>
void test(Vector& vector, const char* name)
{
    set_to_zero(vector);
    typename Vector::value_type ref(0);
	
#if 1	
    vector[2]= value(ref);
    vector[4]= value(ref) + 1.0;
    vector[5]= value(ref) + 2.0;
	
#else // When sparse vectors are used there should be an inserter class for vectors too
    {
		vector::inserter<Vector>  ins(vector);
		ins(2) << value(ref);
		ins(4) << value(ref) + 1.0;
		ins(5) << value(ref) + 2.0;
    }
#endif
	
    cout << "\n\n" << name << "\n";
    cout << "Original vector:\n" << vector << "\n";
	
	// test rscaled_view
    vector::rscaled_view<Vector,double>  rscaled_vector(vector,2.0);
    cout << "vector right scaled with 2.0\n" << rscaled_vector << "\n";
    if (rscaled_vector(2) != rsvalue(ref)) 
		throw "right scaling wrong";
    
    vector::rscaled_view<Vector,ct>  crscaled_vector(vector,ct(0.0, 1.0));
    cout << "vector right scaled with i (complex(0, 1))\n" << crscaled_vector << "\n";
    if (crscaled_vector(2) != crsvalue(ref)) 
		throw "complex right scaling wrong";
	
    cout << "vector right scaled with 2.0 (free function)\n" << rscale(vector,2.0) << "\n";
    if (rscale(vector,2.0)(2) != rsvalue(ref)) 
		throw "right scaling wrong";
	
    cout << "vector right scaled with i (complex(0, 1)) (free function)\n" << rscale(vector,ct(0.0, 1.0)) << "\n";
    if (rscale(vector,ct(0.0, 1.0))(2) != crsvalue(ref)) 
		throw "complex right scaling wrong";
	
	// test divide_by_view
    vector::divide_by_view<Vector,double>  div_vector(vector,0.5);
    cout << "vector divide by 0.5\n" << div_vector << "\n";
    if (div_vector(2) != rsvalue(ref)) 
		throw "divide_by wrong";
    
    vector::divide_by_view<Vector,ct>  cdiv_vector(vector,ct(0.0, -1.0));
    cout << "vector divide by -i (complex(0, -1))\n" << cdiv_vector << "\n";
    if (cdiv_vector(2) != crsvalue(ref)) 
		throw "complex divide_by wrong";
	
    cout << "vector divide by 0.5 (free function)\n" << divide_by(vector,0.5) << "\n";
    if (divide_by(vector,0.5)(2) != rsvalue(ref)) 
		throw "divide_by wrong";
	
    cout << "vector divide by -i (complex(0, -1)) (free function)\n" << divide_by(vector,ct(0.0, -1.0)) << "\n";
    if (divide_by(vector,ct(0.0, -1.0))(2) != crsvalue(ref)) 
		throw "complex divide_by wrong";
	
}



int test_main(int argc, char* argv[])
{
    unsigned size= 7; 
    if (argc > 1) size= atoi(argv[1]); 
	
    dense_vector<double>                                 dv(size);
    dense_vector<complex<double> >                       drc(size);
	
    test(dv, "Dense double vector");
    test(drc, "Dense complex vector");
	
    return 0;
}
