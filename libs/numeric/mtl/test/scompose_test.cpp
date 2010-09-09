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
#include <cmath>
#include <complex>

template <typename T, typename U>
inline bool about(T x, U y)
{
    return std::abs(x - y) < 0.001;
}

template <typename T, typename U>
void test(T x, U y)
{
    using namespace mtl::sfunctor; using mtl::sfunctor::abs;
    using std::cout;

    typedef compose<negate<typename abs<T>::result_type>, abs<T> > nabs;
    cout << "-abs(" << x << ") = " << nabs::apply(x) << "\n";
    if (!about(nabs::apply(x), -std::abs(x)))
	throw "Wrong result for -abs(x)";

    typedef compose<square<typename nabs::result_type>, nabs> snabs;
    cout << "(-abs(" << x << "))^2 = " << snabs::apply(x) << "\n";
    cout << "-std::abs(x) * -std::abs(x) = " << -std::abs(x) * -std::abs(x) << "\n";
    if (!about(snabs::apply(x), -std::abs(x) * -std::abs(x)))
	throw "Wrong result for (-abs(x))^2";
    
    typedef compose_first<plus<typename abs<T>::result_type, U>, abs<T> > plus_abs;
    cout << "abs(" << x << ") + " << y << " = " << plus_abs::apply(x, y) << "\n";
    if (!about(plus_abs::apply(x, y), std::abs(x) + y))
	throw "Wrong result for abs(x) + y";
    
    typedef compose_second<plus<T, typename abs<U>::result_type>, abs<U> > x_plus_abs_y;
    cout << x << " + " << "abs(" << y << ") = " << x_plus_abs_y::apply(x, y) << "\n";
    if (!about(x_plus_abs_y::apply(x, y), x + std::abs(y)))
	throw "Wrong result for x + abs(y)";

    typedef compose_both<plus<T, typename abs<U>::result_type>, negate<T>, abs<U> > minus_x_plus_abs_y;
    cout << "-" << x << " + " << "abs(" << y << ") = " << minus_x_plus_abs_y::apply(x, y) << "\n";
    if (!about(minus_x_plus_abs_y::apply(x, y), -x + std::abs(y)))
	throw "Wrong result for -x + abs(y)";
    
    cout << "l_2(" << x << ", " << 2.0f*x << ") = " << l_2_2D<T>::apply(x, 2.0f*x)  << "\n";
    if (!about(l_2_2D<T>::apply(x, 2.0f*x), std::sqrt(std::abs(5.0f*x*x))))
	throw "Wrong result for l_2_2D(x, 2.0*x)";

    cout << std::endl;
}





int test_main(int argc, char* argv[])
{
    double              a= 3.0, b= -5.0;
    float               c= -9;
    std::complex<float> d(1., 2.);

    test(a, b);
    test(b, a);
    test(c, a);
    test(d, c);

    return 0;
}
