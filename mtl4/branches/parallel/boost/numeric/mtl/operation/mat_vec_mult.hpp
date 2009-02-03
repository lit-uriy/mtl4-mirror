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

#ifndef MTL_MAT_VEC_MULT_INCLUDE
#define MTL_MAT_VEC_MULT_INCLUDE

#include <boost/numeric/mtl/utility/property_map.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/detail/index.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/operation/set_to_zero.hpp>
#include <boost/numeric/mtl/operation/update.hpp>
#include <boost/numeric/mtl/par/dist_mat_cvec_mult.hpp>
#include <boost/numeric/linear_algebra/identity.hpp>



#include <iostream>

namespace mtl { namespace matrix {

#ifdef MTL_HAS_MPI
// Dense matrix vector multiplication
template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
inline void mat_cvec_mult(const Matrix& a, const VectorIn& v, VectorOut& w, Assign, tag::distributed)
{
    dist_mat_cvec_mult(a, v, w, Assign());
}
#endif

// Dense matrix vector multiplication
template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
inline void mat_cvec_mult(const Matrix& a, const VectorIn& v, VectorOut& w, Assign, tag::dense)
{
    // Naive implementation, will be moved to a functor and complemented with more efficient ones

    using math::zero; using mtl::vector::set_to_zero;
    if (size(w) == 0) return;

    if (Assign::init_to_zero) set_to_zero(w);

    typedef typename Collection<VectorOut>::value_type value_type;

    for (unsigned i= 0; i < num_rows(a); i++) {
	value_type tmp= zero(w[i]);
	for (unsigned j= 0; j < num_cols(a); j++) 
	    tmp+= a[i][j] * v[j];
	Assign::update(w[i], tmp);
    }
}

// Multi-vector vector multiplication (tag::multi_vector is derived from dense)
template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
inline void mat_cvec_mult(const Matrix& A, const VectorIn& v, VectorOut& w, Assign, tag::multi_vector)
{
    if (Assign::init_to_zero) set_to_zero(w);
    for (unsigned i= 0; i < num_cols(A); i++)
	Assign::update(w, A.vector(i) * v[i]);
}

// Transposed multi-vector vector multiplication (tag::transposed_multi_vector is derived from dense)
template <typename TransposedMatrix, typename VectorIn, typename VectorOut, typename Assign>
inline void mat_cvec_mult(const TransposedMatrix& A, const VectorIn& v, VectorOut& w, Assign, tag::transposed_multi_vector)
{
    typename TransposedMatrix::const_ref_type B= A.ref; // Referred matrix

    if (Assign::init_to_zero) set_to_zero(w);
    for (unsigned i= 0; i < num_cols(B); i++)
	Assign::update(w[i], dot_real(B.vector(i), v));
}

// Hermitian multi-vector vector multiplication (tag::hermitian_multi_vector is derived from dense)
template <typename HermitianMatrix, typename VectorIn, typename VectorOut, typename Assign>
inline void mat_cvec_mult(const HermitianMatrix& A, const VectorIn& v, VectorOut& w, Assign, tag::hermitian_multi_vector)
{
    typename HermitianMatrix::other::const_ref_type B= A.ref.ref; // Referred matrix

    if (Assign::init_to_zero) set_to_zero(w);
    for (unsigned i= 0; i < num_cols(B); i++)
	Assign::update(w[i], dot(B.vector(i), v));
}


// Sparse matrix vector multiplication
template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
inline void mat_cvec_mult(const Matrix& a, const VectorIn& v, VectorOut& w, Assign, tag::sparse)
{
    smat_cvec_mult(a, v, w, Assign(), typename OrientedCollection<Matrix>::orientation());
}



// Sparse row-major matrix vector multiplication
template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
inline void smat_cvec_mult(const Matrix& a, const VectorIn& v, VectorOut& w, Assign, tag::row_major)
{
    using namespace tag; 
	using mtl::traits::range_generator;  
    using math::zero;
    using mtl::vector::set_to_zero;

    typedef typename range_generator<row, Matrix>::type       a_cur_type;    
    typedef typename range_generator<nz, a_cur_type>::type    a_icur_type;            
	typename mtl::traits::col<Matrix>::type                        col_a(a); 
	typename mtl::traits::const_value<Matrix>::type                value_a(a); 

    if (Assign::init_to_zero) set_to_zero(w);

    typedef typename Collection<VectorOut>::value_type        value_type;

    a_cur_type ac= begin<row>(a), aend= end<row>(a);
    for (unsigned i= 0; ac != aend; ++ac, ++i) {
	value_type tmp= zero(w[i]);
	for (a_icur_type aic= begin<nz>(ac), aiend= end<nz>(ac); aic != aiend; ++aic) 
	    tmp+= value_a(*aic) * v[col_a(*aic)];	
	Assign::update(w[i], tmp);
    }
}

// Sparse column-major matrix vector multiplication
template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
inline void smat_cvec_mult(const Matrix& a, const VectorIn& v, VectorOut& w, Assign, tag::col_major)
{
	using namespace tag; namespace traits = mtl::traits;
	using traits::range_generator;  
	using mtl::vector::set_to_zero;
        typedef typename range_generator<col, Matrix>::type       a_cur_type;             
        typedef typename range_generator<nz, a_cur_type>::type    a_icur_type;            

	typename traits::row<Matrix>::type                        row_a(a); 
	typename traits::const_value<Matrix>::type                value_a(a); 

	if (Assign::init_to_zero) set_to_zero(w);

	unsigned rv= 0; // traverse all rows of v
	for (a_cur_type ac= begin<col>(a), aend= end<col>(a); ac != aend; ++ac, ++rv) {
	    typename Collection<VectorIn>::value_type    vv= v[rv]; 
	    for (a_icur_type aic= begin<nz>(ac), aiend= end<nz>(ac); aic != aiend; ++aic) 
		Assign::update(w[row_a(*aic)], value_a(*aic) * vv);
	}
}

template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
inline void mat_cvec_mult(const Matrix& a, const VectorIn& v, VectorOut& w, Assign as)
{
    mat_cvec_mult(a, v, w, as, typename traits::category<Matrix>::type());
}



}} // namespace mtl::matrix




#endif // MTL_MAT_VEC_MULT_INCLUDE

