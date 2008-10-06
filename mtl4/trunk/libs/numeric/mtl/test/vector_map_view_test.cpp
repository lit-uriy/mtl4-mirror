// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University.
//               2008 Dresden University of Technology and the Trustees of Indiana University.
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#include <iostream>
#include <cmath>
#include <complex>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/vector/dense_vector.hpp>
#include <boost/numeric/mtl/vector/map_view.hpp>
#include <boost/numeric/mtl/operation/print.hpp>
#include <boost/numeric/mtl/operation/set_to_zero.hpp>
#include <boost/numeric/mtl/operation/conj.hpp>
#include <boost/numeric/mtl/operation/scale.hpp>

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

// scaled value
double svalue(double)
{
    return 14.0;
}

ct svalue(ct)
{
    return ct(14.0, 2.0);
}

// conjugated value
double cvalue(double)
{
    return 7.0;
}

ct cvalue(ct)
{
    return ct(7.0, -1.0);
}

// complex scaled value
ct csvalue(double)
{
    return ct(0.0, 7.0);
}

ct csvalue(ct)
{
    return ct(-1.0, 7.0);
}


template <typename Vector>
void test(Vector& vector, const char* name)
{
    set_to_zero(vector);
    typename Vector::value_type ref(0);

    vector[2]= value(ref);
    vector[4]= value(ref) + 1.0;
    vector[5]= value(ref) + 2.0;

#if 0 // When sparse vectors are used there should be an inserter class for vectors too
    {
	vector::inserter<Vector>  ins(vector);
	ins(2) << value(ref);
	ins(4) << value(ref) + 1.0;
	ins(5) << value(ref) + 2.0;
    }
#endif

    cout << "\n\n" << name << "\n";
    cout << "Original vector:\n" << vector << "\n";


    vector::scaled_view<double, Vector>  scaled_vector(2.0, vector);
    cout << "vector  scaled with 2.0\n" << scaled_vector << "\n";
    if (scaled_vector(2) != svalue(ref)) 
	throw "scaling wrong";
    
    vector::conj_view<Vector>  conj_vector(vector);
    cout << "conjugated vector\n" << conj_vector << "\n";
    if (conj_vector(2) != cvalue(ref)) 
	throw " wrong";

    vector::scaled_view<ct, Vector>  cscaled_vector(ct(0.0, 1.0), vector);
    cout << "vector scaled with i (complex(0, 1))\n" << cscaled_vector << "\n";
    if (cscaled_vector(2) != csvalue(ref)) 
	throw "complex scaling wrong";

#if 0 // transposition of vector is not an issue (yet)
    vector::hermitian_view<Vector>  hermitian_vector(vector);
    cout << "Hermitian vector (conjugate transposed)\n" << hermitian_vector << "\n";
    if (hermitian_vector(3, 2) != cvalue(ref)) 
	throw "conjugate transposing  wrong";
#endif

    cout << "vector  scaled with 2.0 (free function)\n" << scale(2.0, vector) << "\n";
    if (scale(2.0, vector)(2) != svalue(ref)) 
	throw "scaling wrong";

    cout << "conjugated vector (free function) \n" << conj(vector) << "\n";
    if (conj(vector)(2) != cvalue(ref)) 
	throw "conjugating wrong";

    cout << "vector scaled with i (complex(0, 1)) (free function)\n" << scale(ct(0.0, 1.0), vector) << "\n";
    if (scale(ct(0.0, 1.0), vector)(2) != csvalue(ref)) 
	throw "complex scaling wrong";


#if 0 // transposition of vector is not an issue (yet)
    cout << "Hermitian  vector (conjugate transposed) (free function)\n" << hermitian(vector) << "\n";
    if (hermitian(vector)(3, 2) != cvalue(ref)) 
	throw "conjugate transposing wrong";
#endif


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
