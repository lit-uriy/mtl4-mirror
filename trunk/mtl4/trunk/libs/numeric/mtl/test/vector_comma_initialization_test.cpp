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
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>


using namespace std;  

template <typename Vector>
void test(Vector& v, const char* name)
{
    std::cout << name << "\n";

    using mtl::Collection;    
    typedef typename Collection<Vector>::value_type   value_type;
    
    v= 7.;
    std::cout << "After v= 7; v == " << v << "\n";
    if (v[2] != value_type(7.))
	throw "Constant assignment wrong";

    v= 8., 45u, -3;
    std::cout << "After v= 8., 45., -3.; v == " << v << "\n";
    if (v[2] != value_type(-3))
	throw "Comma assignment wrong";

    v= 6.;
    std::cout << "After v= 6; v == " << v << "\n";
    if (v[2] != value_type(6.))
	throw "Constant reassignment wrong";

    Vector w(3);
    v= w= 5.;
    std::cout << "After v= w= 5; v == " << v << ", w == " << w << "\n\n";
    if (w[2] != value_type(5.))
	throw "Constant assignment (of w) wrong";
    if (v[2] != value_type(5.))
	throw "Constant reassignment wrong";

#if 0
    v+= 8., 45u, -3;
    std::cout << "After v+= 8., 45., -3.; v == " << v << "\n";
    if (v[0] != value_type(13))
	throw "Comma assignment wrong";
    if (v[2] != value_type(2))
	throw "Comma assignment wrong";
#endif

}



int test_main(int argc, char* argv[])
{
    mtl::dense_vector<float>                  v(3);
    mtl::dense_vector<std::complex<float> >   w(3);

    test(v, "dense_vector<float>");
    test(w, "dense_vector<std::complex<float> >");

    return 0;
}
