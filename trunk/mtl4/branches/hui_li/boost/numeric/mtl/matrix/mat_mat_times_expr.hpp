// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_MAT_MAT_TIMES_EXPR_INCLUDE
#define MTL_MAT_MAT_TIMES_EXPR_INCLUDE

#include <boost/shared_ptr.hpp>
#include <boost/numeric/mtl/matrix/mat_mat_op_expr.hpp>
#include <boost/numeric/mtl/operation/sfunctor.hpp>
#include <boost/numeric/mtl/operation/compute_factors.hpp>


namespace mtl { namespace matrix {

template <typename E1, typename E2>
struct mat_mat_times_expr 
    : public mat_mat_op_expr< E1, E2, sfunctor::times<typename E1::value_type, typename E2::value_type> >,
      public mat_expr< mat_mat_times_expr<E1, E2> >
{
    typedef mat_mat_op_expr< E1, E2, sfunctor::times<typename E1::value_type, typename E2::value_type> > op_base;
    typedef mat_expr< mat_mat_times_expr<E1, E2> >                                                       crtp_base;
    typedef mat_mat_times_expr                   self;
    typedef E1                                   first_argument_type ;
    typedef E2                                   second_argument_type ;
    
    mat_mat_times_expr( E1 const& v1, E2 const& v2 )
	: op_base( v1, v2 ), crtp_base(*this), first(v1), second(v2)
    {}

    // To prevent that cout << A * B prints the element-wise product, suggestion by Hui Li
    typename E1::value_type
    operator()(std::size_t r, std::size_t c) const
    {
	// TBD: Type of matrix product should depend on E1 and E2 
	static boost::shared_ptr<E1> pproduct;

	// If first time compute product
	if (pproduct.get() == 0) {
	    operation::compute_factors<E1, self> factors(*this);
	    pproduct.reset(new E1(num_rows(factors.first), num_cols(factors.second)));
	    *pproduct= factors.first * factors.second;
	}
	return (*pproduct)(r, c);
    }

    first_argument_type const&  first ;
    second_argument_type const& second ;
private:
    // TBD: Type of matrix product should depend on E1 and E2 
    //E1  product;
};


}} // Namespace mtl::matrix

#endif // MTL_MAT_MAT_TIMES_EXPR_INCLUDE
