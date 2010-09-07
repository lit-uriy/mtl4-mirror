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
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/timer.hpp>
#include <boost/mpl/or.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/type_traits/is_integral.hpp>


using namespace std;
using namespace mtl;


int main(int argc, char* argv[])
{
    typedef complex<double>    cd_type;

    const int               rep= 10000;
    double                  n;
    dense_vector<cd_type>   v(10000, cd_type(3.0, 4.0));
    
    boost::timer rtime;
    for (int i= 0; i < rep; i++)
	n= two_norm(v);

    cout << "Run time = " << rtime.elapsed() / rep * 1000 << "ms\n"
	 << "Norm is " << n << "\n";

#if 0
    cout << "d is " << boost::is_floating_point<double>::value << '\n';
    cout << "int is " << boost::is_floating_point<int>::value << '\n';
    cout << "cd is " << boost::is_floating_point<cd_type>::value << '\n';

    cout << "d(f||i) is " << boost::mpl::or_<boost::is_integral<double>, boost::is_floating_point<double> >::value << '\n';
#endif

    return 0;
}
