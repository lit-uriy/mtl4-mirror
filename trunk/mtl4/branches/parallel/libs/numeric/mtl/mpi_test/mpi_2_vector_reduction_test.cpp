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
    using mtl::sum; using mtl::product; using mtl::one_norm;

    for (unsigned i= 0, j= distribution(v).local_to_global(0); i < size(local(v)); i++, j++)
	local(v)[i]= value_type(double(j+1) * pow(-1.0, int(j))); 

    mtl::par::single_ostream sout;
    sout << "\nVector is: " << v << '\n';

    sout << "one_norm(v) = " << one_norm(v) << "\n"; 
    if (one_norm(v) != 15.0) throw "one_norm wrong";

    sout << "one_norm<4>(v) = " << one_norm<4>(v) << "\n";
    if (one_norm<4>(v) != 15.0) throw "one_norm<4> wrong";

    sout << "two_norm(v) = " << two_norm(v) << "\n";
    if (two_norm(v) < 7.4161 || two_norm(v) > 7.4162) throw "two_norm wrong";

    sout << "infinity_norm(v) = " << infinity_norm(v) << "\n"; 
    if (infinity_norm(v) != 5.0) throw "infinity_norm wrong";

    sout << "sum(v) = " << sum(v) << "\n"; 
    if (sum(v) != 3.0) throw "sum wrong";

    sout << "sum<3>(v) = " << sum<3>(v) << "\n";
    if (sum<3>(v) != 3.0) throw "sum<3> wrong";

    sout << "product(v) = " << product(v) << "\n";
    if (product(v) != 120.0) throw "product wrong";

    sout << "product<6>(v) = " << product<6>(v) << "\n";
    if (product<6>(v) != 120.0) throw "product<6> wrong";
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
    test(x, "test double");
    test(xc, "test complex<double>");

#if 0
    dense_vector<float, parameters<row_major> >   ur(5);
    test(ur, "test float in row vector");
#endif

    return 0;
}
 














