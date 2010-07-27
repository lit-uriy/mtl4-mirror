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
#include <complex>
#include <cmath>
#include <boost/test/minimal.hpp>
#include <boost/serialization/complex.hpp>
#include <boost/mpi.hpp>
#include <boost/numeric/mtl/mtl.hpp>

using namespace std;  
namespace mpi = boost::mpi;    

template <typename Vector>
void test(Vector& v, const char* name)
{
    typedef typename mtl::Collection<Vector>::value_type value_type;
    typedef std::complex<value_type>                     complex_type;

    Vector u(5), w(5, value_type(2.0)), x(5, value_type(3.0));


    for (int i= 0, j= distribution(v).local_to_global(0); i < size(local(v)); i++, j++)
	local(v)[i]= value_type(double(j+1) * pow(-1.0, j)); 

    mtl::par::single_ostream sout;
    // sout << name << "\nv is: " << v << '\n';

    // u= v; 
    sout << "u is: " << u << '\n';

    u= value_type(0.0);
    // sout << "u is: " << u << '\n';

    u= v + w;
    // sout << "u is: " << u << '\n';

    u= v + w + x;
    sout << "u= v + w + x is " << u << "\n";
    for (int i= 0, j= distribution(v).local_to_global(0); i < size(local(v)); i++, j++)
	if (std::abs(value_type(double(j+1) * pow(-1.0, j) + 5.0) - u[j]) > 0.001)
	    throw "wrong value in addition";

    u-= 3.0 * w;
    sout << "u-= 3 * w is " << u << "\n";
    for (int i= 0, j= distribution(v).local_to_global(0); i < size(local(v)); i++, j++)
	if (std::abs(value_type(double(j+1) * pow(-1.0, j) - 1.0) - u[j]) > 0.001)
	    throw "wrong value in subtraction";

    u+= dot(v, w) * w + 4.0 * v + 2.0 * w;
    sout << "u+= dot(v, w) * w + 4.0 * v + 2 * w is " << u << "\n";

    sout << "i * v is " << complex_type(0, 1) * v << "\n";
}



int test_main(int argc, char* argv[])
{
    using namespace mtl;
    using mtl::vector::parameters;

    mpi::environment env(argc, argv);

    mtl::vector::distributed<dense_vector<float> >  u(5);
    mtl::vector::distributed<dense_vector<double> > x(5);
    mtl::vector::distributed<dense_vector<std::complex<double> > > xc(5);


    test(u, "test float");
    test(xc, "test complex<double>");
    test(x, "test double");

    return 0;
}
 

