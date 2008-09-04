// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_ORTH_INCLUDE
#define MTL_ORTH_INCLUDE

#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/operation/size.hpp>
#include <boost/numeric/mtl/operation/dot.hpp>
#include <boost/numeric/mtl/operation/two_norm.hpp>

namespace mtl {

    namespace impl {

	template <typename VVector>
	inline void orth(VVector& v, tag::vector)
	{
	    using ::mtl::two_norm;
	    typedef typename mtl::Collection<VVector>::size_type  Size;
	    typedef typename mtl::Collection<VVector>::value_type Vector;
	    typedef typename mtl::Collection<Vector>::value_type  Scalar;

	    for (Size j= 0; j < size(v); ++j) {
		for (Size i= 0; i < j; ++i) {
		    v[j]-= dot(v[i], v[j]) * v[i];
		    std::cout << "i == " << i << ", j == " << j << ", " << v[j]
			      << ", <> == " << dot(v[i], v[j]) << '\n';
		}
		v[j]/= two_norm(v[j]);
	    }
	}

    } // impl



/*! Orthonormalize a vector of vectors.

    The outer type must be a random access collection and
    the vector type must provide a dot function. 
    For instance dense_vector<dense_vector<double> > or
    std::vector<dense_vector<std::complex<double> > > are eligible.
    It is planned to implement the function for matrices as well
    where the columns will be ortho-normalized.
**/
template <typename Value>
inline void orth(Value& value)
{
    return impl::orth(value, typename traits::category<Value>::type());
}


} // namespace mtl

#endif // MTL_ORTH_INCLUDE
