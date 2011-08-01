// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University.
//               2008 Dresden University of Technology and the Trustees of Indiana University.
//               2010 SimuNova UG (haftungsbeschränkt), www.simunova.com.
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#include <iostream>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>

inline bool close(double x, double y) 
{ 
    using std::abs;
    return abs(x - y) < 0.001;
}

int test_main(int , char**)
{
    using namespace std;
    using mtl::lazy;
    
    double                d, rho, alpha= 7.8, beta, gamma;
    const double          cd= 2.6;
    std::complex<double>  z;

    mtl::dense_vector<double> v(6, 1.0), w(6), r(6, 6.0), q(6, 2.0), x(6);
    mtl::dense2D<double>      A(6, 6);
    A= 2.0;
    mtl::compressed2D<double>      B(6, 6);
    B= 2.0;

    (lazy(w)= A * v) || (lazy(d) = lazy_dot(w, v));
    cout << "w = " << w << ", d = " << d << "\n";
    if (!close(d, 12)) throw "wrong dot";    

    (lazy(w)= B * v) || (lazy(d) = lazy_dot(w, v));
    cout << "w = " << w << ", d = " << d << "\n";
    if (!close(d, 12)) throw "wrong dot";

    (lazy(r)-= alpha * q) || (lazy(rho)= lazy_unary_dot(r)); 
    cout << "r = " << r << ", rho = " << rho << "\n";
    if (!close(rho, 552.96)) throw "wrong unary_dot";

    (lazy(x)= 7.0) || (lazy(beta)= lazy_unary_dot(x)); 
    cout << "x = " << x << ", beta = " << beta << "\n";
    if (!close(beta, 294)) throw "wrong unary_dot";
    
    (lazy(x)= 7.0) || (lazy(beta)= lazy_one_norm(x)); 
    cout << "x = " << x << ", beta = " << beta << "\n";
    if (!close(beta, 42)) throw "wrong one_norm";
    
    (lazy(x)= 7.0) || (lazy(beta)= lazy_two_norm(x)); 
    cout << "x = " << x << ", beta = " << beta << "\n";
    if (!close(beta, 17.1464)) throw "wrong two_norm";
    
    (lazy(x)= 7.0) || (lazy(beta)= lazy_infinity_norm(x)); 
    cout << "x = " << x << ", beta = " << beta << "\n";
    if (!close(beta, 7)) throw "wrong one_norm";
    
    (lazy(x)= 7.0) || (lazy(beta)= lazy_sum(x)); 
    cout << "x = " << x << ", beta = " << beta << "\n";
    if (!close(beta, 42)) throw "wrong sum";
    
    (lazy(x)= 7.0) || (lazy(beta)= lazy_product(x)); 
    cout << "x = " << x << ", beta = " << beta << "\n";
    if (!close(beta, 117649)) throw "wrong sum";
    
    (lazy(x)= 2.0) || (lazy(gamma)= lazy_dot(r, x)); 
    cout << "x = " << x << ", gamma = " << gamma << "\n";
    if (!close(gamma, -115.2)) throw "wrong dot";
    
    (lazy(r)= alpha * q) || (lazy(rho)= lazy_dot(r, q)); 
    cout << "r = " << r << ", rho = " << rho << "\n";
    if (!close(rho, 187.2)) throw "wrong dot";
	

    return 0;
}
