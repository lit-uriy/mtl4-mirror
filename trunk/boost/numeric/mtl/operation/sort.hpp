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

#include <algorithm>
#include <boost/numeric/mtl/interface/vpt.hpp>


namespace mtl { namespace vector {

/// sort vector
template <typename Vector>
void inline sort(Vector& x)
{
    std::sort(x.begin(), x.end());
}

/// sort vector with qicksort from lo to hi  
template <typename Vector>  	  	 
void quicksort (Vector& a, typename Collection<Vector>::size_type lo, typename Collection<Vector>::size_type hi)  
{
		vampir_trace<2032> tracer;  	  	 
       	typename Collection<Vector>::size_type i=lo, j=hi;  	  	 
 
        // VergleichsÂ­element x  	  	 
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
	std::cout<< "a=" << a << "\n";
}  	  	 

/// sort vector and permutaion with qicksort from lo to hi  
template <typename Vector, typename PermVec>  	  	 
void quicksort (Vector& a, PermVec& p, typename Collection<Vector>::size_type lo, typename Collection<Vector>::size_type hi)  
{  	  	
		vampir_trace<2033> tracer;
       typename Collection<Vector>::size_type i=lo, j=hi;  	  	 
		
        // VergleichsÂ­element x  	  	 
        typename mtl::Collection<Vector>::value_type x=a[(lo+hi)/2];  
// 	std::cout<< "x=" << x << "\n";
        //  Aufteilung  	  	 
        while (i<=j)  	  	 
        {      	  	 
            while (a[i]<x) i++;   
            while (a[j]>x) j--; 
            
            if (i<=j)  	  	 
            {  	  	 
                swap_row(a, i, j);
		swap_row(p, i, j);
// 		std::cout<< "a=" << a <<"\n";
		 i++; 
  		if( j == 0){
 		  break; 
 		} else {
		  j--;
		}
 		  	
//  		std::cout<< "i=" << i << "\n";
// 		std::cout<< "j=" << j << "\n";
            }  	  	 
//              std::cout<< "j=" << j << "\n";
        }  	
//         std::cout<< "aai=" << i << "\n";
// 	std::cout<< "aaj=" << j << "\n";
// 	std::cout<< "lo=" << lo << "\n";
// 	std::cout<< "hi=" << hi << "\n";
        // Rekursion  	  	 
        if (lo<j) quicksort(a, p, lo, j);  	  	 
        if (i<hi) quicksort(a, p, i, hi);
	
	
	
}  	  	 
 
/// sort vector with permutation
template <typename Vector, typename PermVec>
void inline sort(Vector& x, PermVec& p)
{
//      std::cout<< "x=" << x << "\n";
//      std::cout<< "p=" << p << "\n";
    assert(size(x) == size(p));
    quicksort(x, p, 0, size(x)-1);

}


}} // namespace mtl::vector

#endif // MTL_SORT_INCLUDE