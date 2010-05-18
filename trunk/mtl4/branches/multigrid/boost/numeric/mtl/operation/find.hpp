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

#ifndef MTL_FIND_INCLUDE
#define MTL_FIND_INCLUDE


#include <boost/numeric/mtl/utility/enable_if.hpp>
#include <boost/numeric/mtl/vector/dense_vector.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>

namespace mtl {

    namespace vector {

	/// Find positions of entrys with value tar, Return Vector with Positions and -1 Vector if no mating position was found
	template <typename Vector, typename T>
	dense_vector<unsigned>
	// typename mtl::traits::enable_if_vector<Vector, compressed2D<typename Collection<Vector>::value_type> >::type
	inline find(const Vector& v, const T tar)
	{
	    dense_vector<unsigned>                          find(size(v));
	    find= 0;
	    unsigned   tmp(0);
	std::cout<< "v=" << v << "\n";
	    for (unsigned i= 0; i < size(v); ++i){
		 if (v[i] == tar){
			find[tmp]= i;
			tmp++;
		 }
	    }
		std::cout<< "find=" << find << "\n";
	    if (tmp == 0)
		find[0]= -1;  // no matching
	std::cout<< "find=" << find << "\n";
	  
	    if (tmp <= size(v))
		find.change_dim(tmp+1);	
	std::cout<< "find=" << find << "\n";
	  
	    return find;
	}
    }
} // namespace mtl

#endif // MTL_DIAGONAL_INCLUDE
