// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University.
//               2008 Dresden University of Technology and the Trustees of Indiana University.
//               2010 SimuNova UG (haftungsbeschrÃ¤nkt), www.simunova.com.
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#include <iostream>
#include <boost/numeric/mtl/mtl.hpp>

using namespace std;


template <typename T, typename U>
struct lazy_assign_t
{
    lazy_assign_t(T& first, const U& second) : first(first), second(second) {} 

    T&       first;
    const U& second;

};

template <typename T>
struct lazy_t
{
    lazy_t(T& data) : data(data) {}

    template <typename U>
    lazy_assign_t<T, U> operator=(const U& other) { return lazy_assign_t<T, U>(data, other); }

    T& data;
};

template <typename T>
inline lazy_t<T> lazy(T& x) 
{ return lazy_t<T>(x); }

template <typename T>
inline lazy_t<const T> lazy(const T& x) 
{ return lazy_t<const T>(x); }

#if 0
template <typename Vector1, typename Vector2>
// template <unsigned long Unroll, typename Vector1, typename Vector2, typename ConjOpt>
struct dot_class
{
    // typedef typename detail::dot_result<Vector1, Vector2>::type result_type;
    dot_class(const Vector1& v1, const Vector2& v2) : v1(v1), v2(v2) {}

    // operator result_type() const { return sfunctor::dot<4>::apply(v1, v2, ConjOpt()); }
	    
    const Vector1& v1;
    const Vector2& v2;
};
#endif

template <typename T, typename U>
struct fusion
{
    fusion(const T& first, const U& second) : first(first), second(second) {} 

    const T& first;
    const U& second;
};


template <typename T, typename U>
fusion<T, U> operator||(const T& x, const U& y)
{
    return fusion<T, U>(x, y);
}

int main(int, char**) 
{
    double                d;
    const double          cd= 2.6;
    std::complex<double>  z;

    mtl::dense_vector<double> v(3, 1.0), w(3);
    mtl::dense2D<double>      A(3, 3);
    A= 2.0;

    lazy(d);
    lazy(cd);
    lazy(3. + 4.);

    (lazy(w)= A * v) || (lazy(d) = lazy_dot(w, v));

    return 0;
}
