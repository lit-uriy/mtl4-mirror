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
//
// Algorithm inspired by Nick Vannieuwenhoven, written by Cornelius Steinhardt



#ifndef MTL_MATRIX_ALGORITHMS_INCLUDE
#define MTL_MATRIX_ALGORITHMS_INCLUDE


#include <boost/numeric/itl/pc/value_traits.hpp>
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/pc/element.hpp>
#include <boost/numeric/itl/pc/element_structure.hpp>

#include <iostream>

namespace mtl {
namespace matrix {



/**
 * Construct the sparse data structure.
 */


template< typename ElementStructure, typename Vector> 
mtl::compressed2D<typename ElementStructure::element_type::value_type>*  
assemble_compressed(const ElementStructure& es,	const Vector& order
) {
  	typedef typename ElementStructure::element_type::value_type   value_type;
 	typedef typename ElementStructure::element_iterator           iterator;
 	typedef typename ElementStructure::element_type               element_type;
 	typedef typename element_type::index_type                     index_type;
 	typedef typename element_type::matrix_type                    matrix_type;
	typedef typename matrix_type::size_type                       size_type;
	mtl::compressed2D<value_type> A(es.get_total_vars(),es.get_total_vars());
	set_to_zero(A);
	{//start inserterblock
	  mtl::matrix::inserter<mtl::compressed2D<value_type>, mtl::operations::update_plus<value_type> >  ins(A);
	for(iterator it = es.element_begin(); it != es.element_end(); ++it) {
		element_type& element = *it;
		const index_type& idx = element.get_indices();
		const matrix_type& values = element.get_values();
		for(int i = 0; i < element.nb_vars(); ++i) {
			for(int j = 0; j < element.nb_vars(); ++j) {
				if(values(i,j) != mtl::traits::value_traits<value_type>::zero) {
					ins[order(idx(i))][order(idx(j))] << values(i,j);
				}
			}
		}
	}
	}//end inserterblock

	return new mtl::compressed2D<value_type>(A);
}

}
}//end namespace mtl

#endif // MTL_MATRIX_ALGORITHMS_INCLUDE