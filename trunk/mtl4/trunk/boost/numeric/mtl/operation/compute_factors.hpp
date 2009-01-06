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

#ifndef MTL_COMPUTE_FACTORS_INCLUDE
#define MTL_COMPUTE_FACTORS_INCLUDE

#include <boost/numeric/mtl/mtl_fwd.hpp>

namespace mtl { namespace operation {


// Only defined for matrix::mat_mat_times_expr
template <typename Result, typename Expr>
struct compute_factors {};


// If the two expressions are not products themselves, just refer to the values
template <typename Result, typename E1, typename E2>
struct compute_factors<Result, matrix::mat_mat_times_expr<E1, E2> >
{
    compute_factors(const matrix::mat_mat_times_expr<E1, E2>& src)
	: first(src.first), second(src.second) {}

    const E1& first;
    const E2& second;
};


// First factor is a product itself
// Compute E11 * E12 and store the result in a temporary of type Result
template <typename Result, typename E11, typename E12, typename E2>
struct compute_factors<Result, 
		       matrix::mat_mat_times_expr<matrix::mat_mat_times_expr<E11, E12>, E2> >
{
    compute_factors(const matrix::mat_mat_times_expr<matrix::mat_mat_times_expr<E11, E12>, E2>& src)
	: m11(src.first.first), m12(src.first.second),
	  first(num_rows(m11), num_cols(m12)), second(src.second)
    {
	first= m11 * m12;
    }
    
  private:
    const E11& m11;
    const E12& m12;
  public:
    Result first;
    const E2& second;
};


// Second factor is a product itself
// Compute E21 * E22 and store the result in a temporary of type Result
template <typename Result, typename E1, typename E21, typename E22>
struct compute_factors<Result, 
		       matrix::mat_mat_times_expr<E1, matrix::mat_mat_times_expr<E21, E22> > >
{
    compute_factors(const matrix::mat_mat_times_expr<E1, matrix::mat_mat_times_expr<E21, E22> >& src)
	: first(src.first), m21(src.second.first), m22(src.second.second),
	  second(num_rows(m21), num_cols(m22))
    {
	second= m21 * m22;
    }

    const E1& first;
  private:
    const E21& m21;
    const E22& m22;
  public:
    Result second;
};


// Both factors are products themselves
// Compute E11 * E12 and E21 * E22 and store the results in temporaries of type Result
template <typename Result, typename E11, typename E12, typename E21, typename E22>
struct compute_factors<Result, 
		       matrix::mat_mat_times_expr<matrix::mat_mat_times_expr<E11, E12>,
						  matrix::mat_mat_times_expr<E21, E22> > >
{
    typedef matrix::mat_mat_times_expr<matrix::mat_mat_times_expr<E11, E12>,
				       matrix::mat_mat_times_expr<E21, E22> >    Expr;
    compute_factors(const Expr& src)
	: m11(src.first.first), m12(src.first.second),
	  m21(src.second.first), m22(src.second.second),
	  first(num_rows(m11), num_cols(m12)), 
	  second(num_rows(m21), num_cols(m22))
    {
	first= m11 * m12;
	second= m21 * m22;
    }

  private:
    const E11& m11;
    const E12& m12;
    const E21& m21;
    const E22& m22;
  public:
    Result first, second;
};


// =============================================
// Now the same for element-wise products
// What happens with mixed products? :-!
// =============================================


// If the two expressions are not products themselves, just refer to the values
template <typename Result, typename E1, typename E2>
struct compute_factors<Result, matrix::mat_mat_ele_times_expr<E1, E2> >
{
    compute_factors(const matrix::mat_mat_ele_times_expr<E1, E2>& src)
	: first(src.first), second(src.second) {}

    const E1& first;
    const E2& second;
};


// First factor is a product itself
// Compute E11 * E12 and store the result in a temporary of type Result
template <typename Result, typename E11, typename E12, typename E2>
struct compute_factors<Result, 
		       matrix::mat_mat_ele_times_expr<matrix::mat_mat_ele_times_expr<E11, E12>, E2> >
{
    compute_factors(const matrix::mat_mat_ele_times_expr<matrix::mat_mat_ele_times_expr<E11, E12>, E2>& src)
	: m11(src.first.first), m12(src.first.second),
	  first(num_rows(m11), num_cols(m12)), second(src.second)
    {
	first= ele_prod(m11, m12);
    }
    
  private:
    const E11& m11;
    const E12& m12;
  public:
    Result first;
    const E2& second;
};


// Second factor is a product itself
// Compute E21 * E22 and store the result in a temporary of type Result
template <typename Result, typename E1, typename E21, typename E22>
struct compute_factors<Result, 
		       matrix::mat_mat_ele_times_expr<E1, matrix::mat_mat_ele_times_expr<E21, E22> > >
{
    compute_factors(const matrix::mat_mat_ele_times_expr<E1, matrix::mat_mat_ele_times_expr<E21, E22> >& src)
	: first(src.first), m21(src.second.first), m22(src.second.second),
	  second(num_rows(m21), num_cols(m22))
    {
	second= ele_prod(m21, m22);
    }

    const E1& first;
  private:
    const E21& m21;
    const E22& m22;
  public:
    Result second;
};


// Both factors are products themselves
// Compute E11 * E12 and E21 * E22 and store the results in temporaries of type Result
template <typename Result, typename E11, typename E12, typename E21, typename E22>
struct compute_factors<Result, 
		       matrix::mat_mat_ele_times_expr<matrix::mat_mat_ele_times_expr<E11, E12>,
						  matrix::mat_mat_ele_times_expr<E21, E22> > >
{
    typedef matrix::mat_mat_ele_times_expr<matrix::mat_mat_ele_times_expr<E11, E12>,
				       matrix::mat_mat_ele_times_expr<E21, E22> >    Expr;
    compute_factors(const Expr& src)
	: m11(src.first.first), m12(src.first.second),
	  m21(src.second.first), m22(src.second.second),
	  first(num_rows(m11), num_cols(m12)), 
	  second(num_rows(m21), num_cols(m22))
    {
	first= ele_prod(m11, m12);
	second= ele_prod(m21, m22);
    }

  private:
    const E11& m11;
    const E12& m12;
    const E21& m21;
    const E22& m22;
  public:
    Result first, second;
};




}} // namespace mtl::operation

#endif // MTL_COMPUTE_FACTORS_INCLUDE
