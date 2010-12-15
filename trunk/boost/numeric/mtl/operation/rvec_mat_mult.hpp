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

#ifndef MTL_VECTOR_RVEC_MAT_MULT_INCLUDE
#define MTL_VECTOR_RVEC_MAT_MULT_INCLUDE

namespace mtl { namespace vector {


// Vector sparse matrix multiplication
template <typename VectorIn, typename Matrix, typename VectorOut, typename Assign>
inline void rvec_mat_mult(const VectorIn& v, const Matrix& A, VectorOut& w, Assign, tag::sparse)
{
    rvec_smat_mult(v, A, w, Assign(), typename OrientedCollection<Matrix>::orientation());
}



template <typename VectorIn, typename Matrix, typename VectorOut, typename Assign>
inline void rvec_smat_mult(const VectorIn& v, const Matrix& A, VectorOut& w, Assign, tag::row_major)
{
	using namespace tag; namespace traits = mtl::traits;
	using traits::range_generator;  
	using mtl::vector::set_to_zero;
        typedef typename range_generator<row, Matrix>::type       a_cur_type;             
        typedef typename range_generator<nz, a_cur_type>::type    a_icur_type;            

	typename traits::col<Matrix>::type                        col_a(A); 
	typename traits::const_value<Matrix>::type                value_a(A); 

	if (Assign::init_to_zero) set_to_zero(w);

	unsigned cv= 0; // traverse all columns of v
	for (a_cur_type ac= begin<row>(A), aend= end<row>(A); ac != aend; ++ac, ++cv) {
	    typename Collection<VectorIn>::value_type    vv= v[cv]; 
	    for (a_icur_type aic= begin<nz>(ac), aiend= end<nz>(ac); aic != aiend; ++aic) 
		Assign::update(w[col_a(*aic)], vv * value_a(*aic));
	}
}


}} // namespace mtl::vector

#endif // MTL_VECTOR_RVEC_MAT_MULT_INCLUDE
