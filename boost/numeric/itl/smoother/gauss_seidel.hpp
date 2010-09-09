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

#ifndef ITL_GAUSS_SEIDEL_INCLUDE
#define ITL_GAUSS_SEIDEL_INCLUDE

#include <boost/assert.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/utility/is_row_major.hpp>
#include <boost/numeric/mtl/utility/property_map.hpp>
#include <boost/numeric/mtl/utility/range_generator.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>

namespace itl {

/// Gauss-Seidel smoother
/** Constructor takes references to a matrix and a right-hand side vector.
    operator() is applied on a vector and changes it in place. 
    Matrix must be square, stored row-major and free of zero entries in the diagonal.
    Vectors b and x must have the same number of rows as A. 
**/
template <typename Matrix, typename RHSVector>
class gauss_seidel
{
    typedef typename mtl::Collection<Matrix>::value_type Scalar;
    typedef typename mtl::Collection<Matrix>::size_type  size_type;
  public:
    /// Construct with constant references to matrix and RHS vector
    gauss_seidel(const Matrix& A, const RHSVector& b) : A(A), b(b), dia_inv(num_rows(A)) 
    {
	BOOST_STATIC_ASSERT((mtl::traits::is_row_major<Matrix>::value)); // No CCS
	assert(num_rows(A) == num_cols(A)); // Matrix must be square
	assert(num_cols(A) == size(b));     // Incompatible sizes
	for (size_type i= 0; i < num_rows(A); ++i) {
	    Scalar a= A[i][i];
	    MTL_THROW_IF(a == 0, mtl::missing_diagonal());
	    dia_inv[i]= 1.0 / a;
	}
    }

    /// Apply Gauss-Seidel on vector \p x, i.e. \p x is changed
    template <typename Vector>
    Vector& operator()(Vector& x)
    {
	namespace tag= mtl::tag; using namespace mtl::traits;
	using mtl::begin; using mtl::end; 

        typedef typename range_generator<tag::row, Matrix>::type       a_cur_type;             
        typedef typename range_generator<tag::nz, a_cur_type>::type    a_icur_type;            
	typename col<Matrix>::type                   col_a(A); 
	typename const_value<Matrix>::type           value_a(A); 

	typedef typename mtl::Collection<Vector>::value_type           value_type;

	a_cur_type ac= begin<tag::row>(A), aend= end<tag::row>(A);
	for (unsigned i= 0; ac != aend; ++ac, ++i) {
	    value_type tmp= b[i];
	    for (a_icur_type aic= begin<tag::nz>(ac), aiend= end<tag::nz>(ac); aic != aiend; ++aic) 
		if (col_a(*aic) != i)
		    tmp-= value_a(*aic) * x[col_a(*aic)];	
	    x[i]= dia_inv[i] * tmp;
	}
    }

  private:
    const Matrix&    A;
    const RHSVector& b;
    mtl::dense_vector<Scalar>  dia_inv;
};


} // namespace itl

#endif // ITL_GAUSS_SEIDEL_INCLUDE
