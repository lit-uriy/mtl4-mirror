// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University. 
//               2008 Dresden University of Technology and the Trustees of Indiana University.
//               2010 SimuNova UG, www.simunova.com. 
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also tools/license/license.mtl.txt in the distribution.

// #include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/mtl/vector/dense_vector.hpp>
#include <boost/numeric/mtl/operations.hpp>

// using namespace std;

template <typename VectorU, typename VectorV, typename VectorW>
void test(VectorU& u, VectorV& v, VectorW& , const char* name)
{
    // u= 3.0; v= 4.0; w= 5.0;
    iota(u, -2);

    std::cout << name << ": u = " << u << "\n";
    v = abs(u);
    std::cout << "abs(u) = " << v << "\n";
    MTL_THROW_IF(v[0] != 2.0, mtl::runtime_error("wrong"));
    MTL_THROW_IF(v[4] != 2.0, mtl::runtime_error("wrong"));
    
    
    std::cout << "\n\n";
}
    
    
    
int main(int, char**)
{
    using mtl::vec::parameters;
    using namespace mtl;

    dense_vector<float>   u(5), v(5), w(5);
    dense_vector<double>  x(5), y(5), z(5);
    dense_vector<std::complex<double> >  xc(5), yc(5), zc(5);
    dense_vector<float, parameters<row_major> >   ur(5), vr(5), wr(5);

    std::cout << "Testing vector operations\n";

    test(u, v, w, "test float");
    test(x, y, z, "test double");
    test(u, x, y, "test float, double mixed");
    test(ur, vr, wr, "test float in row vector");
    
    return 0;
}
