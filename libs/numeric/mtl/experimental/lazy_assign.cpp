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
#include <boost/mpl/or.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/numeric/mtl/mtl.hpp>
//#include <boost/numeric/mtl/operation/mat_cvec_times_expr.hpp>

using namespace std;


template <typename T, typename U, typename Assign>
struct lazy_assign_t
{
    typedef Assign  assign_type;

    lazy_assign_t(T& first, const U& second) : first(first), second(second) {} 

    T&       first;
    const U& second;

};

template <typename T, typename U, typename Assign>
void inline evaluate(lazy_assign_t<T, U, Assign>& lazy_assign)
{
    Assign::first_update(lazy_assign.first, lazy_assign.second);
}


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

template <typename T>
struct is_vector_reduction : boost::mpl::false_ {};

template <unsigned long Unroll, typename Vector1, typename Vector2, typename ConjOpt>
struct is_vector_reduction<mtl::vector::dot_class<Unroll, Vector1, Vector2, ConjOpt> >
  : boost::mpl::true_ {};

template <unsigned long Unroll, typename Vector>
struct is_vector_reduction<mtl::vector::unary_dot_class<Unroll, Vector> >
  : boost::mpl::true_ {};


template <typename T>
struct index_evaluatable : boost::mpl::false_ {};

template <typename T, typename U, typename Assign>
struct index_evaluatable<lazy_assign_t<T, U, Assign> >
  : boost::mpl::or_<
      boost::mpl::and_<mtl::traits::is_vector<T>, mtl::traits::is_scalar<U> >,
      boost::mpl::and_<mtl::traits::is_vector<T>, mtl::traits::is_vector<U> >,
      boost::mpl::and_<mtl::traits::is_scalar<T>, is_vector_reduction<U> >
    >
{};

template <typename V1, typename Matrix, typename V2, typename Assign>
struct index_evaluatable<lazy_assign_t<V1, mtl::mat_cvec_times_expr<Matrix, V2>, Assign> >
  : mtl::traits::is_row_major<Matrix> {};





template <typename T, typename U, typename Assign>
typename boost::enable_if<boost::mpl::and_<mtl::traits::is_vector<T>, mtl::traits::is_vector<U> >, mtl::vector::vec_vec_aop_expr<T, U, Assign> >::type
inline index_evaluator(lazy_assign_t<T, U, Assign>& lazy)
{
    return mtl::vector::vec_vec_aop_expr<T, U, Assign>(lazy.first, lazy.second, true);
}

template <typename T, typename U, typename Assign>
typename boost::enable_if<boost::mpl::and_<mtl::traits::is_vector<T>, mtl::traits::is_scalar<U> >, mtl::vector::vec_scal_aop_expr<T, U, Assign> >::type
inline index_evaluator(lazy_assign_t<T, U, Assign>& lazy)
{
    return mtl::vector::vec_scal_aop_expr<T, U, Assign>(lazy.first, lazy.second, true);
}


template <typename Scalar, typename Vector, typename Assign>
struct unary_dot_index_evaluator
{
    unary_dot_index_evaluator(Scalar& scalar, const Vector& v) : scalar(scalar), tmp(0), v(v) {}
    ~unary_dot_index_evaluator() { Assign::apply(scalar, tmp); }
    
    void operator() (std::size_t i) { mtl::vector::two_norm_functor::update(tmp, v[i]); }
    void operator[] (std::size_t i) { (*this)(i); }
    
    Scalar&        scalar;
    Scalar         tmp;
    const Vector&  v;
};

template <typename Scalar, typename Vector, typename Assign>
inline std::size_t size(const unary_dot_index_evaluator<Scalar, Vector, Assign>& eval)
{ return size(eval.v); }

template <typename Scalar, unsigned long Unroll, typename Vector, typename Assign>
unary_dot_index_evaluator<Scalar, Vector, Assign>
inline index_evaluator(lazy_assign_t<Scalar, mtl::vector::unary_dot_class<Unroll, Vector>, Assign>& lazy)
{
    return unary_dot_index_evaluator<Scalar, Vector, Assign>(lazy.first, lazy.second.v);
}

template <typename Scalar, typename Vector1, typename Vector2, typename ConjOpt, typename Assign>
struct dot_index_evaluator
{
    dot_index_evaluator(Scalar& scalar, const Vector1& v1, const Vector2& v2) 
      : scalar(scalar), tmp(0), v1(v1), v2(v2) {}
    ~dot_index_evaluator() { Assign::apply(scalar, tmp); }
    
    void operator() (std::size_t i) { tmp+= ConjOpt()(v1[i]) * v2[i]; }
    void operator[] (std::size_t i) { (*this)(i); }

    Scalar&        scalar;
    Scalar         tmp;
    const Vector1& v1;
    const Vector2& v2;
};

template <typename Scalar, typename Vector1, typename Vector2, typename ConjOpt, typename Assign>
inline std::size_t size(const dot_index_evaluator<Scalar, Vector1, Vector2, ConjOpt, Assign>& eval)
{ 
    return size(eval.v1);
}


template <typename Scalar, unsigned long Unroll, typename Vector1, 
	  typename Vector2, typename ConjOpt, typename Assign>
dot_index_evaluator<Scalar, Vector1, Vector2, ConjOpt, Assign>
inline index_evaluator(lazy_assign_t<Scalar, mtl::vector::dot_class<Unroll, Vector1, Vector2, ConjOpt>, Assign>& lazy)
{
    return dot_index_evaluator<Scalar, Vector1, Vector2, ConjOpt, Assign>(lazy.first, lazy.second.v1, lazy.second.v2);
}


template <typename T, typename U>
struct fusion
{
    template <typename TT, typename UU, typename Assign>
    void check(lazy_assign_t<TT, UU, Assign>& )
    {
	bool vec_scal= boost::mpl::and_<mtl::traits::is_vector<TT>, mtl::traits::is_scalar<UU> >::value;
	bool vec_vec= boost::mpl::and_<mtl::traits::is_vector<TT>, mtl::traits::is_vector<UU> >::value;
	bool scal_red= boost::mpl::and_<mtl::traits::is_scalar<TT>, is_vector_reduction<UU> >::value;
	
	bool ia= boost::mpl::or_<
	    boost::mpl::and_<mtl::traits::is_vector<TT>, mtl::traits::is_scalar<UU> >,
	    boost::mpl::and_<mtl::traits::is_vector<TT>, mtl::traits::is_vector<UU> >,
	    boost::mpl::and_<mtl::traits::is_scalar<TT>, is_vector_reduction<UU> >
	    >::value;
    }


    fusion(T& first, U& second) : first(first), second(second) 
    {
	// check(first); check(second);
	//index_evaluatable<T> it= "";
	//index_evaluatable<U> iu= "";
    }
 
    ~fusion() { eval(index_evaluatable<T>(), index_evaluatable<U>()); }

    template <typename TT, typename UU>
    void eval_loop(TT first_eval, UU second_eval)
    {	
	MTL_DEBUG_THROW_IF(mtl::vector::size(first_eval) != /*mtl::vector::*/  size(second_eval), mtl::incompatible_size());	
	for (std::size_t i= 0, s= size(first_eval); i < s; i++) {
	    first_eval(i);
	    second_eval(i);
	}
    }

    void eval(boost::mpl::true_, boost::mpl::true_)
    {
	cout << "Now I really fuse!\n";
	eval_loop(index_evaluator(first), index_evaluator(second)); 
    }

    template <bool B1, bool B2>
    void eval(boost::mpl::bool_<B1>, boost::mpl::bool_<B2>)
    { evaluate(first); evaluate(second); }

    T& first;
    U& second;
};


template <typename T, typename U>
typename boost::enable_if<boost::mpl::and_<is_lazy<T>, is_lazy<U> >, fusion<T, U> >::type
operator||(const T& x, const U& y)
{
    return fusion<T, U>(const_cast<T&>(x), const_cast<U&>(y));
}

template <typename T, typename U>
typename boost::enable_if<boost::mpl::and_<is_lazy<T>, is_lazy<U> >, fusion<T, U> >::type
fuse(const T& x, const U& y)
{
    return fusion<T, U>(const_cast<T&>(x), const_cast<U&>(y));
}


int main(int, char**) 
{
    double                d, rho, alpha= 7.8, beta, gamma;
    const double          cd= 2.6;
    std::complex<double>  z;

    mtl::dense_vector<double> v(3, 1.0), w(3), r(3, 6.0), q(3, 2.0), x(3);
    mtl::dense2D<double>      A(3, 3);
    A= 2.0;

    (lazy(w)= A * v) || (lazy(d) = lazy_dot(w, v));
    // fuse(lazy(w)= A * v, lazy(d) = lazy_dot(w, v));
    // d= with_reduction(lazy(w)= A * v, lazy_dot(w, v));
    cout << "w = " << w << ", d (6?)= " << d << "\n";


    (lazy(r)-= alpha * q) || (lazy(rho)= lazy_unary_dot(r)); 
    //fuse( lazy(r)-= alpha * q, lazy(rho)= lazy_unary_dot(r) ); 
    // lazy(r)-= alpha * q, lazy(rho)= lazy_unary_dot(r);
    cout << "r = " << r << ", rho (276.48?) = " << rho << "\n";

    (lazy(x)= 7.0) || (lazy(beta)= lazy_unary_dot(x)); 
    cout << "x = " << x << ", beta (147?) = " << beta << "\n";
    
    (lazy(x)= 2.0) || (lazy(gamma)= lazy_dot(r, x)); 
    cout << "x = " << x << ", gamma (-57.6?) = " << gamma << "\n";
    
    (lazy(r)= alpha * q) || (lazy(rho)= lazy_dot(r, q)); 
    cout << "r = " << r << ", rho (93.6?) = " << rho << "\n";

    return 0;
}
