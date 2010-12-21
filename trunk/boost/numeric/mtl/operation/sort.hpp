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

#ifndef MTL_SORT_INCLUDE
#define MTL_SORT_INCLUDE

#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/operation/swap_row.hpp>

namespace mtl { namespace vector {

/// sort vector
template <typename Vector>
void inline sort(Vector& x)
{
	Vector tmp(x);
	std::cout << "tmp=" << tmp << "\n";
	x= quicksort(tmp, 0, size(tmp)-1);
	
	std::cout<<"x="<< x << "\n";
    //return x;
}

/// sort vector with qicksort from lo to hi
template <typename Vector>
Vector quicksort (Vector a, typename Collection<Vector>::size_type lo, typename Collection<Vector>::size_type hi)
    {
        typename Collection<Vector>::size_type i=lo, j=hi;

        // Vergleichs­element x
        typename mtl::Collection<Vector>::value_type x=a[(lo+hi)/2];
	    //  Aufteilung
        while (i<=j)
        {    
            while (a[i]<x) i++; 
            while (a[j]>x) j--;
            if (i<=j)
            {
                swap_row(a, i, j);
                i++; j--;
            }
        }

        // Rekursion
        if (lo<j) quicksort(a, lo, j);
        if (i<hi) quicksort(a, i, hi);
		return a;
    }




} // namespace vector

} // namespace mtl

#endif // MTL_SORT_INCLUDE
