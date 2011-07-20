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
#include <boost/mpl/bool.hpp>
#include <boost/mpl/and.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/numeric/mtl/mtl.hpp>

using namespace std;


template <typename T, typename U, typename Assign>
struct lazy_assign_t
{
    typedef Assign  assign_type;

    lazy_assign_t(T& first, const U& second) : first(first), second(second) {} 

    T&       first;
    const U& second;

};

template <typename T>
struct is_lazy : boost::mpl::false_ {};

template <typename T, typename U, typename Assign>
struct is_lazy<lazy_assign_t<T, U, Assign> > : boost::mpl::true_ {};


template <typename T>
struct lazy_t
{
    lazy_t(T& data) : data(data) {}

    template <typename U>
    lazy_assign_t<T, U, mtl::assign::assign_sum> operator=(const U& other) 
    { return lazy_assign_t<T, U, mtl::assign::assign_sum>(data, other); }

    template <typename U>
    lazy_assign_t<T, U, mtl::assign::plus_sum> operator+=(const U& other) 
    { return lazy_assign_t<T, U, mtl::assign::plus_sum>(data, other); }

    template <typename U>
    lazy_assign_t<T, U, mtl::assign::minus_sum> operator-=(const U& other) 
    { return lazy_assign_t<T, U, mtl::assign::minus_sum>(data, other); }

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
typename boost::enable_if<boost::mpl::and_<is_lazy<T>, is_lazy<U> >, fusion<T, U> >::type
operator||(const T& x, const U& y)
{
    return fusion<T, U>(x, y);
}

int main(int, char**) 
{
    double                d, rho, alpha= 7.8;
    const double          cd= 2.6;
    std::complex<double>  z;

    mtl::dense_vector<double> v(3, 1.0), w(3), r(3, 6.0), q(3, 2.0);
    mtl::dense2D<double>      A(3, 3);
    A= 2.0;

    lazy(d);
    lazy(cd);
    lazy(3. + 4.);

    (lazy(w)= A * v) || (lazy(d) = lazy_dot(w, v));
    (lazy(r)-= alpha * q) || (lazy(rho)= lazy_unary_dot(r)); 

    return 0;
}
