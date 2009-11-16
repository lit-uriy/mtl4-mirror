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

#ifndef MTL_STD_OUTPUT_OPERATOR_INCLUDE
#define MTL_STD_OUTPUT_OPERATOR_INCLUDE

#ifdef MTL_HAS_STD_OUTPUT_OPERATOR

#include <iostream>
#include <utility>
#include <vector>

namespace std {

    /// Print standard vector
    /** Only available when compiled with macro flag MTL_HAS_STD_OUTPUT_OPERATOR
	to avoid (reduce) conflicts with other libraries. **/
    template <typename T>
    inline ostream& operator<< (ostream& os, vector<T> const&  v)
    {
	os << '[';
	for (size_t r = 0; r < v.size(); ++r)
	    os << v[r] << (r < v.size() - 1 ? "," : "");
	return os << ']';
    }

    /// Print standard pair
    /** Only available when compiled with macro flag MTL_HAS_STD_OUTPUT_OPERATOR
	to avoid (reduce) conflicts with other libraries. **/
    template <typename T, typename U>
    inline ostream& operator<< (ostream& os, pair<T, U> const& p)
    {
	return os << '(' << p.first << ',' << p.second << ')';
    }
} // namespace mtl

#endif // MTL_HAS_STD_OUTPUT_OPERATOR

#endif // MTL_STD_OUTPUT_OPERATOR_INCLUDE
