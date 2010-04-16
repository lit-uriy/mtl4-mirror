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


#if 0
template <typename Vector>
void test(Vector& v, const char* name)
{
    mtl::par::single_ostream sout;
    // sout << name << "\nv is: " << v << '\n';


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
#endif


int test_main(int argc, char* argv[])
{
    using namespace mtl;
    using mtl::vector::parameters;

    mpi::environment env(argc, argv);
    boost::mpi::communicator world;


    if (world.size() != 4) {
	std::cerr << "Example works only for 4 processors!\n";
	env.abort(87);
    };

    typedef mtl::vector::parameters<mtl::tag::col_major, mtl::vector::fixed::dimension<8> > dimension8;
    typedef mtl::dense_vector<double, dimension8>                                           vector8;

    typedef mtl::matrix::parameters<mtl::tag::row_major, mtl::index::c_index, mtl::fixed::dimensions<3, 3> > mparams3x3;
    typedef mtl::dense2D<double, mparams3x3>                                                                 matrix3x3;

#if 0
    mtl::vector::distributed<mtl::dense_vector<double> >         v(5);
    mtl::vector::distributed<vector8>                            X;

    mtl::matrix::distributed<mtl::matrix::dense2D<double>  >     a(2, 2);
    mtl::matrix::distributed<matrix3x3  >                        b(2, 2);
#endif

    mtl::dense_vector<double>         v(5);
    vector8                           X;

    mtl::matrix::dense2D<double>      a(2, 2);
    matrix3x3                         b(3, 3);

    if (world.rank() == 0) {

	X[0]=0.1; X[1]=0.45; X[2]=0.6; X[3]=1.23;X[4]=9.4;X[5]=7.; X[6]=0.8;X[7]=12.7;

	v[0]=1.; v[1]=2.; v[2]=3.; v[3]=4.; v[4]=5.;

	a[0][0]=0.1;a[0][1]=0.2;a[1][0]=0.3; a[1][1]=0.4;

	b[0][0]=21.;b[0][1]=22.; b[0][2]=23.;b[1][0]=24.; b[1][1]=25.;b[1][2]=26.; b[2][0]=27.; b[2][1]=28.; b[2][2]=29.;
    };


    boost::mpi::broadcast(world, X, 0);
    boost::mpi::broadcast(world, v, 0);
#if 1
    boost::mpi::broadcast(world, a, 0);
    boost::mpi::broadcast(world, b, 0);
#endif


    std::cout << "Process #" << world.rank() << " has X " << X << std::endl;
    std::cout << "Process #" << world.rank() << " has v " << v << std::endl;
#if 1
    std::cout << "Process #" << world.rank() << " has matrix a \n" << a << std::endl;
    std::cout << "Process #" << world.rank() << " has matrix b \n" << b << std::endl;
#endif

    return 0;
}
 

