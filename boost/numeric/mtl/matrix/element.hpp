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
 *  Created on: Oct 2, 2009
 *      Author: heazk
 */

 
#ifndef MTL_ELEMENT_INCLUDE
#define MTL_ELEMENT_INCLUDE

#include <algorithm>
#include <iostream>
#include <set>

#include <boost/unordered_set.hpp>

#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/matrix/dense2D.hpp>
#include <boost/numeric/mtl/vector/dense_vector.hpp>
#include <boost/numeric/mtl/utility/make_copy_or_reference.hpp>
#include <boost/numeric/itl/pc/comparators.hpp>


namespace mtl { namespace matrix {

///  A class representing an element with ValType equalsT he type of the numeric values
template <typename ValType>
class element 
{

public:
    /// The type of this element.
    typedef element<ValType> element_type;

    /// The value type of the matrix and rhs elements.
    typedef ValType value_type;

    /// The type of a set of neighbours.
    typedef std::vector<element_type*> neighbour_collection_type; 

    /// An iterator over the neighbours of this element.
    typedef typename neighbour_collection_type::iterator neighbour_iterator;

    /// The type of an unordered set of neighbours.
    typedef typename boost::unordered_set<
	element_type*,
	compare::address_hasher<element_type>,
	compare::address_compare_equal<element_type>
       > neighbour_set_type;

    /// The type of the iterator over an unorderd set of neighbours.
    typedef typename neighbour_set_type::iterator neighbour_set_iterator_type;

    /// The type of matrix.
    typedef mtl::matrix::dense2D<value_type> matrix_type;

    /// The type of the index vector.
    typedef mtl::vector::dense_vector<int> index_type; 


/*******************************************************************************
 * Constructors
 ******************************************************************************/


    /**
     * Constructs the element using the memory specified by the two references.
     *
     * p_indices: 	a reference to the memory where the indices may be stored.
     * p_values:	a reference to the memory where the values may be stored.
     */
  public:
    element(int p_sequence_number, const index_type& p_indices, const matrix_type& p_values) 
      : m_indices(p_indices),	
	m_values(p_values),
	m_sequence_number(p_sequence_number)
    {}

    element() 
      :	m_sequence_number(-1) {}

    element(const element_type& other) 
      :	m_sequence_number(-1)
    {	*this = other;	}

    /// Deep copy the given element.
    void operator=(const element_type& other) 
    {
	m_sequence_number = other.m_sequence_number;
	m_neighbours = other.m_neighbours;

	m_indices = other.m_indices;
	m_values =  other.m_values;
    }

    /// Returns the unique identifier of this element.
    inline int get_id() const {	return m_sequence_number; }
    inline int& get_id()      {return m_sequence_number; }

    /// Returns the number of variables.
    inline int nb_vars() const { return size(m_indices); }

    /// Returns the number of values.
    inline int nb_values() const { return nb_vars()*nb_vars(); }

    /// Reference to the matrix of values.
    inline matrix_type& get_values() { return m_values; }

    /// Constant reference to the matrix of values.
    inline const matrix_type& get_values() const { return m_values; }

    /// Reference to the indices.
    inline const index_type& get_indices() const { return m_indices; }

    /// Mutable reference to the indices.
    inline index_type& get_indices() {	return m_indices; }

    /// The actual number of non-zero values.
    int nnz() const 
    {
	const value_type    zero= math::zero(value_type());
	int nbr_nz = 0;
	for (int r = 0; r < nb_vars(); ++r) 
	    for (int c = 0; c < nb_vars(); ++c) 
		nbr_nz +=  get_values()(r,c) != zero;
	return nbr_nz;
    }

    /// Reference to the set of neighbours.
    neighbour_collection_type& get_neighbours() { return m_neighbours;	}

    /// Reference to the set of neighbours.
    const neighbour_collection_type& get_neighbours() const { return m_neighbours; }

    /// Number of neighbours this element is connected to.
    int get_nb_neighbours() const {	return int(m_neighbours.size()); }


/*******************************************************************************
 * Useful Inspector Methods
 ******************************************************************************/

    /// The set of nodes that is incident to the element.
    boost::unordered_set<int> get_incident_nodes() const 
    {
	boost::unordered_set<int> nodes(2 * get_nb_neighbours());
	for(typename neighbour_collection_type::const_iterator neigh_it = m_neighbours.begin();
	    neigh_it != m_neighbours.end(); ++neigh_it) {
	    element_type& neigh = **neigh_it;
	    nodes.insert(neigh.get_indices().begin(), neigh.get_indices().end());
	}
	// Remove the nodes of the element.
	for( int i = 0; i < nb_vars(); ++i ) 
	    nodes.erase( get_indices()(i) );
	return nodes;
    }


    /// Get the set of level-k neighbours, for a given k. 
    neighbour_set_type get_level_neighbours(const int level = 1) 
    {
	neighbour_set_type result( get_nb_neighbours() * level );

	if (level > 0) {
	    result.insert( m_neighbours.begin(), m_neighbours.end() );
	    if (level > 1) {
		for(int i = 0; i < get_nb_neighbours(); ++i) {
		    neighbour_set_type neighs(m_neighbours[i]->get_level_neighbours(level-1));
		    result.insert( neighs.begin(), neighs.end() );
		}
		result.erase( this );
	    }
	}
	return result;
    }

/*******************************************************************************
 * Manipulation
 ******************************************************************************/
  public:
    /// Permutes the rows and the columns of the element coefficient matrix along 
    /// with the indices such that the latter are sorted in ascending order.
    void sort_indices() {
	if(size(m_indices) == 0) {
	    assert(size(m_values) == 0);
	    return;
	}

	bool sorted = true;
	for(int i = 0; i < nb_vars()-1; ++i) {
	    sorted &= (get_indices()(i) < get_indices()(i+1));
	}
	if(sorted) {
	    return;
	}

	index_type  orig_index( get_indices() );
	matrix_type orig_matrix( get_values() );

	std::sort(
		  &(get_indices()(0)),
		  &(get_indices()(0))+nb_vars()
		  );

	index_type orig_offset( nb_vars() );
	orig_offset = -1;
	for(int i = 0; i < nb_vars(); ++i) {
	    int seek_idx = get_indices()(i);
	    int j = 0;
	    for(; (j < nb_vars()) && (orig_index(j) != seek_idx); ++j){};
	    orig_offset(i) = j;
	}

	matrix_type& values = get_values();
	for(int r = 0; r < nb_vars(); ++r) {
	    for(int c = 0; c < nb_vars(); ++c) {
		values(r,c) = orig_matrix( orig_offset(r), orig_offset(c) );
	    }
	}

#ifndef NDEBUG
	sorted = true;
	for(int i = 0; i < nb_vars()-1; ++i) {
	    sorted &= (get_indices()(i) < get_indices()(i+1));
	}
	assert(sorted);
#endif
    }


  public:
    /// Removes the given set of nodes from the element
    template< class Vector >
    void remove_nodes(const Vector& nodes, element_type& el) {
	if(size(m_indices) == 0) {
	    assert(size(m_values) == 0);
	    return;
	}
	if(nb_vars() == 0) {
	    return;
	}

#ifndef NDEBUG
	bool sorted = true;
	for(unsigned int i = 1; i < mtl::size(nodes); ++i) {
	    sorted &= ( nodes[i-1] < nodes[i] );
	}
	assert(sorted);
#endif

	const int nb_nodes = mtl::size(nodes);

	// Count number of remaining variables.
	int new_nb_nodes = nb_vars();
	{
	    int i = 0, j = 0;
	    while( i < nb_vars() && j < nb_nodes ) {
		const int diff = get_indices()(i) - nodes[j];
		if( diff < 0 ) {
		    ++i;
		} else if( diff > 0 ) {
		    ++j;
		} else {
		    --new_nb_nodes;
		    ++i;
		    ++j;
		}
	    }
	}
	assert(new_nb_nodes >= 0);

	// Construct new index array.
	index_type index;
	index_type local_index(new_nb_nodes);
	if(new_nb_nodes > 0) {
	    index.change_dim(new_nb_nodes);
//	    index = new index_type(new_nb_nodes);
	    int i = 0, j = 0, pos = 0;
	    while( i < nb_vars() && j < nb_nodes ) {
		const int diff = get_indices()(i) - nodes[j];
		if( diff < 0 ) {
		    assert( pos < new_nb_nodes );
	//	    (*index)(pos) = get_indices()(i);
		    index[pos] = get_indices()(i);
		    local_index(pos) = i;
		    ++pos;
		    ++i;
		} else if( diff > 0 ) {
		    ++j;
		} else {
		    ++i;
		    ++j;
		}
	    }
	    while( i < nb_vars() ) {
		assert( pos < new_nb_nodes );
//		(*index)(pos) = get_indices()(i);
		index[pos] = get_indices()(i);
		local_index(pos) = i;
		++pos;
		++i;
	    }
	} else {
//	    index = new index_type(0);
	}

	matrix_type values;
	if(new_nb_nodes > 0) {
	    values.change_dim( new_nb_nodes, new_nb_nodes );
	    matrix_type tmp(get_values()), tmp2(new_nb_nodes, new_nb_nodes);
	    for(unsigned int i=0;i<size(local_index);i++){
		for(unsigned int j=0;j<size(local_index);j++){
		    tmp2[i][j]=tmp[local_index(i)][local_index(j)];
		}
	    }
	    values = tmp2;
	} else {
	std::cout<< "ELSE\n";
//	    values = new matrix_type(0,0);
	}
	// Update the neighbourhood.
	std::set<int> remove_neighs;
	for(
	    neighbour_iterator neigh_it = m_neighbours.begin();
	    neigh_it != m_neighbours.end();
	    ++neigh_it
	    ) {
	    element_type& neigh = **neigh_it;

	    // Search a matching index.
	    bool connected = false;
	    {
		int i = 0, j = 0;
		while(
		      (i < new_nb_nodes) &&
		      (j < neigh.nb_vars()) &&
		      !connected
		      ) {
//		    const int diff = (*index)(i) - neigh.get_indices()(j);
		    const int diff = index[i] - neigh.get_indices()(j);
		    if(diff < 0) {
			++i;
		    } else if(diff > 0) {
			++j;
		    } else {
			connected = true;
		    }
		}
	    }

	    // If not found, then remove ourself from the neighbours and vice versa.
	    if(!connected) {
		neighbour_iterator pos = std::find(neigh.get_neighbours().begin(), neigh.get_neighbours().end(), this );
		if( (pos != neigh.get_neighbours().end()) && (&neigh != &el) ) {
		    neigh.get_neighbours().erase(pos);
		}
		remove_neighs.insert( neigh.get_id() );
	    }
	}

	// Remove the neighbours we're no longer connected to.
	for(std::set<int>::iterator it = remove_neighs.begin(); it != remove_neighs.end(); ++it) {
	    const int seek_seq_nbr = *it;
	    for (std::size_t j = 0; j < m_neighbours.size(); ++j) 
		if (m_neighbours[j] != 0 && m_neighbours[j]->get_id() == seek_seq_nbr) {
		    m_neighbours.erase( m_neighbours.begin()+j );
		    break;
		}
	}

	if(new_nb_nodes == 0) {
	    m_neighbours.clear();
	}

	m_indices.change_dim(0);
	m_values.change_dim(0, 0);
	m_indices = index;
	m_values = values;
    }

    /// Absorbs the values of the given matrix with the given index.
    template< class Matrix, class Vector >
    void absorb(Matrix& other_values, Vector& other_indices) 
    {
	const value_type    zero= math::zero(value_type());
#ifndef NDEBUG
	bool sorted = true;
	for(unsigned int i = 1; i < size(other_indices); ++i) {
	    sorted &= ( other_indices(i-1) < other_indices(i) );
	}
	assert(sorted);
#endif
		
	const int other_idx_size = size( other_indices );

	// Determine set of common indices.
	const int max_common_idx =
	    (nb_vars() < other_idx_size) ? nb_vars() : other_idx_size;
	mtl::vector::dense_vector<int> my_idx( max_common_idx );
	mtl::vector::dense_vector<int> ot_idx( max_common_idx );
	int offset = 0;
	for(int i = 0, j = 0; i < nb_vars() && j < other_idx_size; ) {
	    int diff = (get_indices()(i) - other_indices(j));
	    if(diff == 0) {
		my_idx(offset) = i;
		ot_idx(offset) = j;
		++offset;
		++i;
		++j;
	    } else if(diff < 0) {
		++i;
	    } else {
		++j;
	    }
	}

	// Absorb the values.
	for(int i = 0; i < offset; ++i) {
	    for(int j = i; j < offset; ++j) {
		get_values()( my_idx(i), my_idx(j) ) += other_values( ot_idx(i), ot_idx(j) );
		other_values( ot_idx(i), ot_idx(j) ) = zero;
		get_values()( my_idx(j), my_idx(i) ) += other_values( ot_idx(j), ot_idx(i) );
		other_values( ot_idx(j), ot_idx(i) ) = zero;
	    }
	}
    }
    
    /// Removes the numerical values from the element.
    void clear() {
	m_neighbours.clear();
	m_neighbours.resize(1);
	matrix_type empty;
	swap(m_values, empty);
	m_indices.change_dim(0);
    }

    template< class ValueType > friend class element_structure;

  private:
    /// The set of neighbours of the element.
    neighbour_collection_type m_neighbours;

    /// The set of indices of this element.
    index_type m_indices;

    /// The [Size x Size] element matrix.
    matrix_type m_values;

    /// A unique sequence number for the element, indicating it's order relative to other elements.
    int m_sequence_number;

    int *dummy;
};


/// Print an element to an output stream.
template<typename OStream, class ValueType>
OStream& operator<<(OStream& out, element<ValueType>& el) 
{
    out << "ID: " << el.get_id() << "\n";
    if(el.nb_vars() > 0) {
	out << "Indices: (" << el.get_indices()(0);
	for(int i = 1; i < el.nb_vars(); ++i) {
	    out << ", " << el.get_indices()(i);
	}
	out << ")\n";
    } else {
	out << "Indices: ()\n";
    }
    out << "Neighbours: (";
    if(el.nb_vars() > 0) {
	for(int i = 0; i < el.get_nb_neighbours(); ++i) {
	    out << el.get_neighbours()[i]->get_id() << ", ";
	}
    }
    out << ")\n";
    if(el.nb_vars() > 0) {
	out << "Values: \n";
	for(int i = 0; i < el.nb_vars(); ++i) {
	    out << "[ ";
	    for(int j = 0; j < el.nb_vars(); ++j) {
		out << el.get_values()(i,j) << " ";
	    }
	    out << "]\n";
	}
    } else {
	out << "Values: []\n";
    }
    return out;
}

  }} // mtl::matrix

#endif // MTL_ELEMENT_INCLUDE