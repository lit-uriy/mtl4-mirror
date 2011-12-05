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

/*
 * Implementation of the IMF factorization and application routines.
 */
 
#ifndef MTL_IMF_ALGORITHMS_INCLUDE
#define MTL_IMF_ALGORITHMS_INCLUDE

#include <iostream>
#include <boost/numeric/mtl/matrix/compressed2D.hpp>
#include <boost/numeric/mtl/matrix/coordinate2D.hpp>
#include <boost/numeric/mtl/utility/make_copy_or_reference.hpp>

#include <string.h>
#include <boost/numeric/itl/pc/binary_heap.hpp>
#include <boost/numeric/itl/pc/sorting.hpp>

#include <vector>
#include <map>
#include <set>
#include <complex>
#include <limits>

#include "boost/unordered_map.hpp"
#include "boost/unordered_set.hpp"

#include <boost/numeric/mtl/utility/irange.hpp>

namespace itl {   namespace pc { 
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
// INTRUSIVE HEAP HELPER FUNCTORS
//
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

namespace heap {
/**
 * An information extending structure for elements to support the reduction
 * process.
 */
template<class Element, class Key>
struct HeapInformation {
	Key m_key;
	Element* m_parent;
	Element* m_left_child;
	Element* m_right_child;

	HeapInformation() :
		m_key(0.0), m_parent(0), m_left_child(0), m_right_child(0) {}
};

/**
 * A functor returning a reference to the degree of an element.
 */
template<class Element, class Key>
struct GetKey {
	Key& operator()(Element* element) const {
		return static_cast<HeapInformation<Element,Key>* >(
			element->get_extra_pointer()
		)->m_key;
	}
};

/**
 * A functor returning a reference to the parent of an element (in a heap).
 */
template<class Element, class Key>
struct GetParent {
	Element*& operator()(Element* element) const {
		return static_cast<HeapInformation<Element,Key>* >(
			element->get_extra_pointer()
		)->m_parent;
	}
};

/**
 * A functor returning a reference to the left child of an element (in a heap).
 */
template<class Element, class Key>
struct GetLeftChild {
	Element*& operator()(Element* element) const {
		return static_cast<HeapInformation<Element,Key>* >(
			element->get_extra_pointer()
		)->m_left_child;
	}
};

/**
 * A functor returning a reference to the right child of an element (in a heap).
 */
template<class Element, class Key>
struct GetRightChild {
	Element*& operator()(Element* element) const {
		return static_cast<HeapInformation<Element,Key>* >(
			element->get_extra_pointer()
		)->m_right_child;
	}
};

/**
 * Compares two elements based on their degrees.
 */
template<class Element, class Key>
struct DegreeCompare {
	/**
	 * Compares two elements based on their degree. Ties are broken by the
	 * sequence numbers of the elements.
	 */
	inline bool operator()(Element* first, Element* second) const {
		const int seq_fst = first->get_id();
		const int seq_snd = second->get_id();

		const Key key_fst = static_cast<HeapInformation<Element,Key>* >(
				first->get_extra_pointer()
			)->m_key;
		const Key key_snd = static_cast<HeapInformation<Element,Key>* >(
				second->get_extra_pointer()
			)->m_key;

		if (key_fst == key_snd) {
			return seq_fst < seq_snd;
		}
		return key_fst < key_snd;
	}
};

} // end namespace heap


enum Status { UNMARKED, DIAGONAL, MARKED_CURRENT, REMOVED, NON_DIAGONAL };


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
// DIAGONAL BLOCK SELECTION PRIORITY ESTIMATORS
//
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/**
 * Estimates the priority based on the number of nodes the element is connected
 * to for the purpose of the local Schur complement computation. A higher
 * priority is given if the element is connected to fewer nodes.
 */
template< class Element, class NodeStatusVector, bool UseStatus = false >
struct MinConnectedNodesEstimation {
	inline int operator()(
			const Element& el,
			const NodeStatusVector& status
	) const {
		typedef typename Element::neighbour_collection_type neigh_type;

		// Determine set of all nodes.
		std::vector<int> nodes;
		const neigh_type& neighs = el.get_neighbours();
		for(int i = 0; i < el.get_nb_neighbours(); ++i) {
			nodes.insert(
				nodes.end(),
				neighs[i]->get_indices().begin(),
				neighs[i]->get_indices().end()
			);
		}

//		radix_sort( &nodes[0], nodes.size() );   // TODO INCLUDE RADIX_SORT
		std::sort( nodes.begin(), nodes.end() );

		int degree = -el.nb_vars()+1;
		for(unsigned int i = 1; i < nodes.size(); ++i) 
			if( nodes[i-1] != nodes[i] )
			    if (!UseStatus || status[nodes[i]] == UNMARKED)
				++degree;
		return degree;
	}
};



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
// IMF Factorization
//
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


template<class Type>
struct AddressHasher {
	long operator()(const Type*& object) const {
		return reinterpret_cast<long>( object );
	}
};

template<class Element, class StatusVector>
struct IsRemoved {
	const StatusVector& m_status;
	IsRemoved(const StatusVector& status) : m_status(status) { }

	bool operator()(const Element *const el) const {
		return m_status[ el->get_id() ] == REMOVED;
	}
};

/**
 * Constructs the EBE-ML-ILU preconditioner from the given mesh. The mesh is
 * altered in the process.
 */
 template< typename ValType >
template<  class Mesh >
void itl::pc::imf_preconditioner<ValType>::factor(const Mesh& mesh , const int maxlofi   
) {
	typedef typename Mesh::element_type element_type;
	typedef typename Mesh::element_iterator element_iterator;
	typedef typename element_type::value_type value_type;
	typedef typename element_type::neighbour_collection_type neigh_coll_type;
	typedef typename element_type::neighbour_iterator neigh_iterator;
	typedef typename element_type::neighbour_set_type neigh_set_type;
	typedef typename element_type::neighbour_set_iterator_type neigh_set_iterator;
	typedef typename neigh_coll_type::const_iterator const_neigh_iterator;
	typedef typename element_type::index_type index_type;
	typedef typename element_type::matrix_type matrix_type;
	
	typedef typename mtl::matrix::coordinate2D<value_type> coo_sparse_type_upper;
	typedef typename mtl::matrix::coordinate2D<value_type> coo_sparse_type_lower;
		     

	typedef std::map<int, int> cmap;
	typedef typename cmap::iterator cmap_iterator;

	typedef value_type key_type;
	typedef utils::binary_heap<
		element_iterator,
		key_type,
		heap::DegreeCompare<element_type, key_type>,
		element_type*,
		heap::GetKey<element_type, key_type>,
		heap::GetParent<element_type, key_type>,
		heap::GetLeftChild<element_type, key_type>,
		heap::GetRightChild<element_type, key_type>
	> my_heap;


	// Constants.
	const int nb_elements = mesh.get_total_elements();
	const int nb_vars = mesh.get_total_vars();
	const int UNPERMUTED = -1;
	
	// A DS tracking the set of elements during the construction.
	//	Invariant: elements[i]->get_sequence_number() == i
	//	Invariant: elements[i] == 0  iff  element[i] is removed
	std::vector<element_type*> elements;
	elements.reserve( nb_elements + (nb_elements >> 4) );
	{
		element_iterator it = mesh.element_begin();
		for(int i = 0; i < nb_elements; ++i) {
			elements.push_back( *&it );
			++it;
		}
	}

	// Data structures for the preconditioner.
	std::vector<element_type*> block_diagonal;
	std::vector<coo_sparse_type_upper*> upper_matrices;
	std::vector<coo_sparse_type_lower*> lower_matrices;
	std::vector<int> diagonal_offsets;
	diagonal_offsets.push_back(0);
	m_ordering = UNPERMUTED;

	////////////////////////////////////////////////////////////////////////////
	// Auxiliary data structures.
	////////////////////////////////////////////////////////////////////////////

		// Binary heap containing the dense elements that should still be
		// considered for selection on the diagonal of the current level.
		heap::DegreeCompare<element_type, key_type> key_compare;
		heap::GetKey<element_type, key_type> get_key;
		heap::GetParent<element_type, key_type> get_parent;
		heap::GetLeftChild<element_type, key_type> get_left;
		heap::GetRightChild<element_type, key_type> get_right;
		my_heap unmarked_elements(
				key_compare,
				get_key,
				get_parent,
				get_left,
				get_right
		);
		std::vector< heap::HeapInformation<element_type, key_type>* >	red_info( nb_elements );

		// Another data structure for the elements that should be considered
		// for the diagonal
		std::vector<element_type*> unmarked_elements_srtd;
		std::vector<int          > unmarked_elements_degr;
		unmarked_elements_srtd.reserve( nb_elements );
		unmarked_elements_degr.reserve( nb_elements );

		// A DS to keep track of whether the element is on the diagonal, not
		// on the diagonal or not yet marked.
		std::vector<Status> el_status(nb_elements, UNMARKED);

		// A DS to keep track of the variables that have been forced into the
		// reduced system.
		std::vector<Status> in_reduced( mesh.get_total_vars(), UNMARKED );

	////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////

	// Degree estimators.
	const MinConnectedNodesEstimation<element_type, std::vector<Status> > min_nodes_estimate =
	      MinConnectedNodesEstimation<element_type, std::vector<Status> >();

	std::cout << mesh.get_total_vars() << " nodes remaining" << std::endl;

	int level = 1;
	int perm_off = 0;
	int perm_high = 0;
	int block_diag_low = 0;
	int max_sequence_number = nb_elements;

 	do {

		block_diag_low = block_diagonal.size();

/***************************************************************************
 * Phase 0: Determine priority of unmarked elements
 **************************************************************************/

		// Make sure the set of unmarked elements is empty again.
		assert( unmarked_elements.empty() );		
		unmarked_elements_degr.clear();
		unmarked_elements_srtd.clear();
		// Construct the set of available diagonal elements along with their
		// degrees.
		for( unsigned int i = 0; i < elements.size(); ++i ) {
			// Skip the element if it no longer exists, or it is a diagonal
			// element.
			if( (el_status[i] == REMOVED) || (el_status[i] == DIAGONAL) ) {
				continue;
			}

			element_type& el = *(elements[i]);
			// Reset the status.
			el_status[i] = UNMARKED;

			int degree = min_nodes_estimate(el, in_reduced);
		        unmarked_elements_degr.push_back( degree );
			unmarked_elements_srtd.push_back( &el );
		}

/***************************************************************************
 * Phase 1: Select the block diagonal elements
 **************************************************************************/
/********************************SKIP UPDATE***********************************/

		// Sort the candidate diagonal elements by their degree in ascending
		// order.
		sort_along<int, element_type*>(
			&*(unmarked_elements_degr.begin()),
			&*(unmarked_elements_srtd.begin()),
			unmarked_elements_degr.size()
		);

		// For each of the candidate diagonal elements ...
		for(unsigned int i = 0; i < unmarked_elements_srtd.size(); ++i) {

			// Select the minimum element.
			element_type& el = *(unmarked_elements_srtd[i]);
			// If the element is already marked, skip it.
			if( el_status[el.get_id()] != UNMARKED ) {
				continue;
			}

			// If the element is unmarked, add it to the set of diagonal
			// elements

			// Add the element to the diagonal.
			block_diagonal.push_back(&el);
			// Mark the element.
			el_status[el.get_id()] = DIAGONAL;

			// Update permutation vector
			for(int i = 0; i < el.nb_vars(); ++i) {
				m_ordering( el.get_indices()(i) ) = perm_off;
				++perm_off;
				assert(perm_off <= mesh.get_total_vars());
			}

			// Determine level-2 neighbours.
			neigh_set_type lvl2_neighs = el.get_level_neighbours( 2 );

			// Mark all level-2 neighbours.
			for(
				neigh_set_iterator neigh_it = lvl2_neighs.begin();
				neigh_it != lvl2_neighs.end();
				++neigh_it
			) {
				element_type& neigh = **neigh_it;
				assert( el_status[neigh.get_id()] != DIAGONAL );
				el_status[ neigh.get_id() ] = MARKED_CURRENT;
			}
		}


/*******************************UPDATE DEGREE**********************************/
		// Update diagonal block offset.
		diagonal_offsets.push_back( perm_off );
		//save upperbound for number of L and U entrys
		unsigned int upperbound(0);
		for(unsigned int i=0;i< block_diagonal.size();i++){
			mtl::dense_vector<int> involve_node(block_diagonal[i]->get_indices());
			for(unsigned int j=0;j< block_diagonal[i]->get_neighbours().size();j++){
			  mtl::dense_vector<int> involve_neigh(block_diagonal[i]->get_neighbours()[j]->get_indices());
			  unsigned int c(0);
			  for(unsigned int a= 0; a < size(involve_node); a++){
			      for(unsigned int b= 0; b < size(involve_neigh); b++){
				if(involve_node[a] == involve_neigh[b])
				  c++;			    
			      }
			  }
			  upperbound+= c*(size(involve_neigh)-c);
			}
		}
			
		// Update permutation offsets.
// 		perm_low = perm_high;
		perm_high = perm_off;
		
/***************************************************************************
 * Phase 2: Compute the update matrices
 **************************************************************************/
		coo_sparse_type_upper* L =	new coo_sparse_type_upper(nb_vars, nb_vars, upperbound);
		coo_sparse_type_lower* U =	new coo_sparse_type_lower(nb_vars, nb_vars, upperbound);
		lower_matrices.push_back(L);
		upper_matrices.push_back(U);
		// For each diagonal block element ... (in parallel)
		unsigned int ku= 0, kl= 0;
		for(std::size_t b_i = block_diag_low; b_i < block_diagonal.size();++b_i ) {
			element_type& diag_el = *(block_diagonal[b_i]);
			// Copy the level-1 neighbours.
			neigh_coll_type& diag_neighs = diag_el.get_neighbours();

			// Determine the set of incident nodes.
			boost::unordered_set<int> diag_incident_nodes =
				diag_el.get_incident_nodes();
			index_type& p = diag_el.get_indices();
			index_type q(
				diag_incident_nodes.size() == 0 ?
					1 : diag_incident_nodes.size()
			);
			typename boost::unordered_set<int>::const_iterator it =
				diag_incident_nodes.begin();
			for(unsigned int i = 0; i < diag_incident_nodes.size(); ++i) {
				q(i) = *it;
				++it;
			}
			const int n1 = size(p);
			const int n2 = diag_incident_nodes.size();
			sort(q);
			assert(n1 > 0);

			// Construct a mapping from global to local node numbers.
			cmap to_local;
			for(int i = 0; i < n1; ++i) {
				to_local[p(i)] = i;
			}
			
			for(int i = 0; i < n2; ++i) {
				to_local[q(i)] = mtl::size(p) + i;
			}
			// Construct the frontal matrix.
			block_type frontal( n1+n2, n1+n2 );
			frontal = mtl::traits::value_traits<value_type>::zero;

			// For each connected neighbour, add their values to the frontal
			// matrix.
			for(neigh_iterator neigh_it = diag_neighs.begin(); neigh_it != diag_neighs.end(); ++neigh_it) {
				element_type& neigh = **neigh_it;
				assert( el_status[neigh.get_id()] != DIAGONAL );
				assert( el_status[neigh.get_id()] != REMOVED );

				// Remap indices.
				index_type local_idx( neigh.get_indices() );
				
				for(int i = 0; i < neigh.nb_vars(); ++i) {
					local_idx(i) = to_local[ local_idx(i) ];
				}
				//insert connectet neighbour
				{
				  mtl::matrix::inserter<mtl::dense2D<value_type>, mtl::operations::update_plus<value_type> > ins(frontal);
				  ins << element_matrix(neigh.get_values(), local_idx, local_idx);
				}
				neigh.get_values() = mtl::traits::value_traits<value_type>::zero;
			}
			// Add the values of the diagonal element to the frontal matrix.
			{
				// Remap indices.
				index_type local_idx( diag_el.get_indices() );
				for(int i = 0; i < diag_el.nb_vars(); ++i) {
					local_idx(i) = to_local[ local_idx(i) ];
				}
				//insert the diagonal element
				{	
				  mtl::matrix::inserter<mtl::dense2D<value_type>, mtl::operations::update_plus<value_type> > ins(frontal);
				  ins << element_matrix(
				  diag_el.get_values(), local_idx, local_idx);
				}
				diag_el.get_values() = mtl::traits::value_traits<value_type>::zero;
			}
			//
			// Store the L and U part.
			//

#if 0 //TEST with compressed2D matrix, but slow				
		//	mtl::compressed2D<value_type>  L(nb_vars,nb_vars), U(nb_vars,nb_vars);  //TODO a littel bit to much
			{ //Start Inserter Block
	//		    mtl::matrix::inserter<mtl::compressed2D<value_type> >   insL(L);  
	//		    mtl::matrix::inserter<mtl::compressed2D<value_type> >   insU(U);
				mtl::matrix::inserter<mtl::compressed2D<value_type> >  insL(*L);
				mtl::matrix::inserter<mtl::compressed2D<value_type> >  insU(*U);
				
			for(int i = 0; i < n1; ++i) { // p-part
				for(int j = n1; j < n1+n2; ++j) { // q-part
					// Assumption: structural symmetry.
					if(frontal(i,j) != traits::value_traits<value_type>::zero) {
 						insU[p(i)][q(j-n1)] << frontal(i,j);
					}
					if(frontal(j,i) != traits::value_traits<value_type>::zero){
 						insL[q(j-n1)][p(i)] << frontal(j,i);
					}
				}
			}

			}//End Inserter Block
			
			
#endif
			
			for(int i = 0; i < n1; ++i) { // p-part
				for(int j = n1; j < n1+n2; ++j) { // q-part
				  // Assumption: structural symmetry.
					if(frontal(i,j) != mtl::traits::value_traits<value_type>::zero) {
						U->insert(p(i),q(j-n1),frontal(i,j), ku );
						ku++;
					}
					if(frontal(j,i) != mtl::traits::value_traits<value_type>::zero){
						L->insert(q(j-n1),p(i),frontal(j,i), kl );
						kl++;
					}
				}
			}

			mtl::dense2D<value_type> D(n1,n1),C(n1,n1);
			mtl::irange  n0(0,n1);
			// Invert the D-part matrix.
			C=frontal[n0][n0];
//  			D=inv(frontal[n0][n0]);  //doesnot work in current version
			D=inv(C);
			// Store the D part.
    			diag_el.get_values() = D ;
			frontal[n0][n0] = diag_el.get_values();
		
	
 			if( (level <= maxlofi) && (n2 > 0) ) { 
//------------------------------------------------------------------------------
//  					ELEMENT COALESCING APPROACH
//------------------------------------------------------------------------------
				// The lofi is below the user-requested level. Use Algorithm 2
				// in [1].
				// Compute the Schur complement.
				mtl::irange n1r(0,n1), n2r(n1, mtl::imax); 
				frontal[n2r][n2r]-= frontal[n2r][n1r] * frontal[n1r][n1r] * frontal[n1r][n2r];
				mtl::irange nz(n1,n1+n2);
				matrix_type Z( frontal[nz][nz] );
				// Construct the new element.
				element_type* fill = new element_type(max_sequence_number, q, Z);
				// Add processing information to the required DSs.
				elements.push_back( fill );
				el_status.push_back( UNMARKED );

			
				// Determine the level-2 neighbours.
				neigh_set_type lvl2_neighs = diag_el.get_level_neighbours( 2 );
				// Remove the level-1 neighbours of the diagonal element.
				for(neigh_iterator it = diag_neighs.begin(); it != diag_neighs.end(); ++it) {
					element_type& neigh = **it;
					neigh.clear();
					el_status[neigh.get_id()] = REMOVED;
				}

				// Update the neighbourhood of the level-2 neighbours.
				const IsRemoved<element_type, std::vector<Status> > is_removed( el_status );
				for(neigh_set_iterator it = lvl2_neighs.begin(); it != lvl2_neighs.end(); ++it) {
					element_type& neigh = **it;

					// Skip the level-1 neighbours (removed) and the diagonal
					// element.
					if(
						(el_status[neigh.get_id()] == DIAGONAL) ||
						(el_status[neigh.get_id()] == REMOVED)
					) {
						continue;
					}
					// The element is in the strict level-2 neighbourhood.
					// Remove the level-1 neighbours from its set of neighbours.
					neigh_coll_type& neigh_neighs = neigh.get_neighbours();
					neigh_iterator new_end = std::remove_if(
						neigh_neighs.begin(), neigh_neighs.end(), is_removed
					);
					neigh_neighs.erase(new_end, neigh_neighs.end());

					// Add the newly generated element as neighbour, and vice
					// versa.
					neigh_neighs.push_back( fill );
					fill->get_neighbours().push_back( &neigh );
					
				}
				// Update sequence number.
				++max_sequence_number;
			} else if (n2 > 0) {

//------------------------------------------------------------------------------
//  					ELEMENT DISTRIBUTION APPROACH
//------------------------------------------------------------------------------
				// The lofi is above the user-requested level. Use Algorithm 3
				// in [1] to compute the approximate Schur complement element-
				// wise.

				// Distribute the values of the generalized element across the
				// level-2 neighbours of the diagonal element.

				// Remove the nodes of the diagonal element from its
				// neighbours.
				for(neigh_iterator neigh_it = diag_neighs.begin(); neigh_it != diag_neighs.end(); ++neigh_it)
				{
					element_type& neigh = **neigh_it;
					neigh.remove_nodes( diag_el.get_indices(), diag_el );

					assert( el_status[neigh.get_id()] != DIAGONAL );
					assert( el_status[neigh.get_id()] != REMOVED );

					// If the element is entirely removed, mark it as such.
					if( neigh.nb_vars() == 0 ) {
						el_status[neigh.get_id()] =	REMOVED;
					}
				}

				// Compute the Schur complement.
				mtl::irange n1r(0,n1), n2r(n1, mtl::imax); 
				frontal[n2r][n2r]-= frontal[n2r][n1r] * frontal[n1r][n1r] * frontal[n1r][n2r];
				// Distribute the values of the (modified) update matrix
				// over the level-1 neighbours.
				mtl::irange nz(n1,n1+n2);
  				matrix_type S( frontal[nz][nz] );
				for(neigh_iterator neigh_it = diag_neighs.begin(); neigh_it != diag_neighs.end();++neigh_it) {
					element_type& el = **neigh_it;

					if( el_status[el.get_id()] != REMOVED ) {
						el.absorb(S, q);
					}
				}

				// Do not distribute over the entire level-2 neighbourhood.
				// This is expensive, while not adding much in terms of quality
				// of the approximation.

			} else {
				// There are no more nodes in the Schur complement, but it could
				// be that some elements other than the diagonal element overlap
				// completely with said element.

				// In this case, simply remove all neighbours.
				for(
					neigh_iterator neigh_it = diag_neighs.begin();
					neigh_it != diag_neighs.end();
					++neigh_it
				) {
					element_type& neigh = **neigh_it;
					neigh.remove_nodes( diag_el.get_indices(), diag_el );

					assert( el_status[neigh.get_id()] != DIAGONAL );
					assert( el_status[neigh.get_id()] != REMOVED );

					// The element is now entirely removed, mark it as such.
					assert( neigh.nb_vars() == 0 );
					el_status[neigh.get_id()] = REMOVED;
				}

			} // END ELEMENT DISTRIBUTION
			// Clear the neighbourhood of the diagonal element.
			diag_el.get_neighbours().clear();
		}

		std::cout << "level " << level << " complete: ";
		std::cout << mesh.get_total_vars()-perm_high << " nodes remaining."<< "\n";
		
		++level;
	} while( (mesh.get_total_vars() - perm_high > 0) );	// There are still variables to cover.
	
		
	diagonal_offsets.push_back( mesh.get_total_vars() );
	assert( mesh.get_total_vars() - perm_high == 0 );
	assert( lower_matrices.size()+2 == diagonal_offsets.size() );
	/***************************************************************************
	 * Phase 3: Apply permutation vector to lower and upper matrices
	 **************************************************************************/
	mtl::matrix::traits::permutation<>::type P(permutation(m_ordering));
  
	for( std::size_t k = 0; k < lower_matrices.size(); ++k ) {
 		coo_sparse_type_lower& L = *(lower_matrices[k]);
 		coo_sparse_type_upper& U = *(upper_matrices[k]);
 		if (size(L.value_array())> 0){
		    L.compress(); U.compress();
		    mtl::dense_vector<unsigned int> &L_row(L.row_index_array()), 
						    &L_col(L.column_index_array()), 
						    &U_row(U.row_index_array()), 
						    &U_col(U.column_index_array());
		    const int off_low = diagonal_offsets[k];
		    for(unsigned int i = 0; i < size(L_row); ++i) {
			L_row[i] = m_ordering( L_row[i] );
			L_col[i] = m_ordering( L_col[i] ) - off_low;
		    };
		    for(unsigned int i = 0; i < size(U_row); ++i) {
			  U_row[i] = m_ordering( U_row[i] ) - off_low;
			  U_col[i] = m_ordering( U_col[i] );
		    }
		    L.sort_col();
		    U.sort_row();
		}
		sparse_type_lower* lower = new sparse_type_lower( L );
		m_lower.push_back(lower);
		sparse_type_upper* upper = new sparse_type_upper( U );
		m_upper.push_back(upper);

#if 0 //compressed2D try
		if(off_low > 0){ 
		  mtl::matrix::traits::permutation<>::type P_col(permutation(col_perm));
 		  L_t= trans(P) * L * P * trans(P_col);
 		  U_t= P_col * trans(P) * U * P ;
		}else {
 		  L_t= trans(P) * L * P;  
 		  U_t= trans(P) * U * P;
		}
		sparse_type_lower* lower = new sparse_type_lower( L_t );
		m_lower.push_back(lower);
		sparse_type_upper* upper = new sparse_type_upper( U_t );
		m_upper.push_back(upper);
#endif
		
	}
	/***************************************************************************
	 * Phase 4: Construct the IMF preconditioner
	 **************************************************************************/
	// Copy the block diagonal values into a consecutive array.
	m_diagonal = new matrix_type[ block_diagonal.size() ];
	for(std::size_t i = 0; i < block_diagonal.size(); ++i) {
		assert( el_status[block_diagonal[i]->get_id()] == DIAGONAL );
 		unsigned int diarows(num_rows(block_diagonal[i]->get_values()));
  		m_diagonal[i].change_dim(diarows,diarows);
		m_diagonal[i] = block_diagonal[i]->get_values();
	}
	// Copy the diagonal offsets to a consecutive array.
	int* diagonal_index = new int[ diagonal_offsets.size() ];
	memcpy(
		diagonal_index,
		&(diagonal_offsets[0]),
		sizeof(int)*diagonal_offsets.size()
	);
	assert( diagonal_offsets.back() == mesh.get_total_vars() );
	assert( lower_matrices.size()+2 == diagonal_offsets.size() );

	m_levels = lower_matrices.size();
	m_nb_blocks = block_diagonal.size();
	m_diagonal_index = diagonal_index;
#if 0
	long total_mem = 0;
	long diag_nnz = 0;
	for(unsigned int i = 0; i < m_nb_blocks; ++i) {
		diag_nnz += block_diagonal[i]->nnz();
		total_mem += block_diagonal[i]->nb_values();
	}
	long total_nnz = 0;
	for(unsigned int i = 0; i < m_lower.size(); ++i) {
		total_nnz += ( *(m_lower[i]) ).nnz() ;
	}
	for(unsigned int i = 0; i < m_upper.size(); ++i) {
		total_nnz += ( *(m_upper[i]) ).nnz() ;
	}
	total_mem += total_nnz;
	total_nnz += diag_nnz;

	//ONLY more information about approximation
	double pct_mem_increase = 100*double(total_mem - orig_nnz) / orig_nnz;
	double pct_diag = 100*double(diag_nnz) / total_mem;

 	unsigned int off = settings.max_lofi+1; 
	if( off >= diagonal_offsets.size() ) {
		off = diagonal_offsets.size()-1;
	}
	int unity_eigs = diagonal_index[off];
	double pct_unity_eigs = 100*double(unity_eigs) / nb_vars;

	std::cout << "\n----------IMF STATISTICS--------\n";
 	std::cout << "Level of fill-in:       " << sets.max_lofi << "\n";
	std::cout << "Original non-zeros:     " << orig_nnz << "\n";
	std::cout << "Total nnz (actual):     " << total_nnz << "\n";
	std::cout << "Total nnz (structural): " << total_mem << "\n";
	std::cout << "Nb levels:              " << level << "\n";
	std::cout << "Eigenvalues unity:      " << unity_eigs << " (" <<
			pct_unity_eigs << "%)\n";
	std::cout << "Increase in memory:     " << pct_mem_increase << "%\n";
	std::cout << "Non-zeros on diagonal:  " << pct_diag << "%\n";
	std::cout << "----------------------------------\n" << std::endl;
#endif

}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
// IMF APPLICATION
//
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


/**
 * Applies the IMF preconditioner to a matrix. The contents of the rhs vector
 * are overwritten.
 *
 * The algorithm is implemented without recursion, as was already suggested by
 * Y. Saad in [2].
 */

template< class ValType >
template< class Vector >
Vector imf_preconditioner<ValType>::imf_apply(const Vector& rhs) const
{
	using namespace mtl;
	Vector res(rhs);
	// Forward elimination.
	{
	  int b_off = 0;
	  for(int level = 0; level < m_levels; ++level) {
		const int off_low = m_diagonal_index[level];
		const int off_high = m_diagonal_index[level+1];
		const int n1 = off_high - off_low;
		assert(off_low <= off_high);
		// Compute dy = inv(D)*y
		vector_type dy(n1);
		dy = mtl::traits::value_traits<ValType>::zero;  
		for(int off = off_low; off < off_high; ++b_off ) { //parallel
			const int block_size = num_rows( m_diagonal[b_off] );
			dy[mtl::irange(off-off_low, off-off_low+block_size)] = m_diagonal[b_off] * res[mtl::irange(off, off + block_size)];
			off += block_size;
		}
		// Compute x = x - E*dy
		assert( m_lower[level] );
		Vector big(num_cols(*m_lower[level])); big=0;
		big[mtl::irange(0,n1)] = dy;
 		res -= ( *(m_lower[level]) * big );
	  }
	}
	// Backward elimination.
	{
	int b_off = m_nb_blocks-1;
	for(int level = m_levels-1; level >= 0; --level) {

		const int off_low = m_diagonal_index[level];
		const int off_high = m_diagonal_index[level+1];
		// y' = y - Fx
		assert( m_upper[level] );
		
		vector_type yp( *(m_upper[level]) * res );

		res[mtl::irange(off_low, off_high) ] -= yp[mtl::irange(0, off_high-off_low) ];
		// y = inv(D)*y'
		for(int off = off_high; off > off_low; --b_off ) {
			const int block_size = num_rows(m_diagonal[b_off]);
			assert(b_off >= 0);
			assert(off-block_size >= off_low);

			vector_type dy(m_diagonal[b_off] * res[irange(off-block_size, off)] );
			res[irange(off-block_size, off)] = dy;
			off -= block_size;
		}
	}
	assert(b_off == -1);
	}
    return res;
}


} // end namespace pc
} // end namespace itl


#endif // MTL_IMF_ALGORITHMS_INCLUDE
