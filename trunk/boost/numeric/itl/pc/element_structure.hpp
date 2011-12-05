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



#ifndef MTL_ELEMENT_STRUCTURE_INCLUDE
#define MTL_ELEMENT_STRUCTURE_INCLUDE

#include <iostream>
#include <ostream>


#include <boost/numeric/itl/pc/element.hpp>
#include <boost/numeric/mtl/mtl.hpp>

namespace mtl {

namespace print {

	template< class Type >
	struct print_type {
		template<class Stream>
		static void print(Stream&);
	};

	template< >
	struct print_type<double> {
		template<class Stream>
		static void print(Stream& str) {
			str << "double\n";
		}
	};

	template< >
	struct print_type<float> {
		template<class Stream>
		static void print(Stream& str) {
			str << "double\n";
		}
	};

	template< >
	struct print_type<std::complex<double> > {
		template<class Stream>
		static void print(Stream& str) {
			str << "complex\n";
		}
	};

	template< class Type >
	struct print_value {
		template<class Stream>
		static void print(Stream&);
	};

	template< >
	struct print_value<double> {
		template<class Stream>
		static void print(Stream& str, double& val) {
			str << std::scientific << std::setprecision(15);
			str << val;
		}
	};

	template< >
	struct print_value<float> {
		template<class Stream>
		static void print(Stream& str, float& val) {
			str << std::scientific << std::setprecision(15);
			str << double(val);
		}
	};

	template< >
	struct print_value<std::complex<double> > {
		template<class Stream>
		static void print(Stream& str, std::complex<double>& val) {
			str << std::scientific << std::setprecision(15);
			str << val.real() << "\t" << val.imag() << "\t";
		}
	};
}

/**
 * A generic abstract base class for meshes. It describes the concept of a mesh.
 */
template< class ValueType >
class element_structure 
{

    /*******************************************************************************
     * Type Definitions
     ******************************************************************************/

  public:
    /**
     * The type of the numerical values of the element coefficient matrices.
     */
    typedef ValueType value_type;

    /**
     * The type of the element.
     */
    typedef element<value_type> element_type;

    /**
     * The type of index arrays.
     */
    typedef typename element_type::index_type index_type;

    /**
     * The type of the iterator over the elements of the mesh.
     */
    typedef element_type* element_iterator;

    /**
     * The type of this class.
     */
    typedef element_structure<ValueType> this_type;
    

    /*******************************************************************************
     * Constructors
     ******************************************************************************/


    /**
     * Standard constructor.
     */
  public:
    element_structure(int total_elements, int total_vars, element_type* elements)
      : m_total_elements(total_elements),
	m_total_vars(total_vars),
	m_elements(elements)
    { }

    /**
     * Copy the given mesh.
     */
  public:
    element_structure(this_type const& other)
      :
      m_total_elements(other.m_total_elements),
      m_total_vars(other.m_total_vars),
      m_elements(
		 other.m_total_elements == 0 ?
		 0 :
		 new element_type[other.get_total_elements()]
		 )
    {
	typedef typename element_type::neighbour_collection_type neigh_coll_type;

	int j = 0;
	bool ordered = true;
	for(
	    element_iterator it = other.element_begin();
	    it != other.element_end();
	    ++it
	    ) {
	    // Deep copy the elements.
	    m_elements[j] = *it;
	    ordered &= (it->get_id() == j);
	    ++j;
	}
	assert( ordered );
	// Reconstruct the network of neighbours.
	for(
	    element_iterator it = this->element_begin();
	    it != this->element_end();
	    ++it
	    ) {
	    neigh_coll_type new_neighs;
	    neigh_coll_type& old_neighs = it->get_neighbours();
	    for(int i = 0; i < it->get_nb_neighbours(); ++i) {
		element_type& neigh = *(old_neighs[i]);
		int pos = neigh.get_id();
		new_neighs.push_back( this->m_elements+pos );
	    }
	    old_neighs.assign(new_neighs.begin(), new_neighs.end());
	}
    }


    /*******************************************************************************
     * Destructors
     ******************************************************************************/


    /**
     * Default destructor.
     */
  public:
    ~element_structure() {
	if(m_elements) {
	    delete[] m_elements;
	    m_elements = 0;
	}
    }
    
    ///assumption elements with quadratic elementmatrix
    template< class Vector >
    Vector operator*(  Vector& x) const {
	Vector m_tmp(size(x), 0.0);
  	for(int elmi= 0; elmi < m_total_elements; elmi++){
	   unsigned int n(size(m_elements[elmi].get_indices()));
	    for( unsigned int i= 0; i < n; i++){
	        for( unsigned int j= 0; j < n; j++){
		    m_tmp[m_elements[elmi].get_indices()[i]]+= m_elements[elmi].get_values()[i][j]*x[m_elements[elmi].get_indices()[j]];
	        }
	     }
 	}
	return m_tmp;
	}


    /*******************************************************************************
     * Inspector Members
     ******************************************************************************/
  public:

    /**
     * Returns the total number of elements in the grid.
     */
    int get_total_elements() const {
	return m_total_elements;
    }

    /**
     * Returns the total number of variables.
     */
    int get_total_vars() const {
	return m_total_vars;
    }

    /**
     * Returns the total number of non-zero values.
     */
    int get_total_nnz() const {
	int nnz = 0;
	for(element_iterator it = element_begin(); it != element_end(); ++it) {
	    nnz += it->nnz();
	}
	return nnz;
    }

    /**
     * Returns an iterator to the first element.
     */
    element_iterator element_begin() const {
	return m_elements + 0;
    }

    /**
     * Returns an iterator to the element past the last element.
     */
    element_iterator element_end() const {
	return m_elements + this->get_total_elements();
    }

    /*******************************************************************************
     * File Operations
     ******************************************************************************/
  public:

    /**
     * Writes the elements to the specified file.
     *
     * TODO test this code
     */
    void write_to_file(const std::string& filename) 
    {
	using namespace print;

	std::ofstream file(filename.c_str());

	// Write header information.
	file << get_total_elements() << "\n";
	file << this->get_total_vars() << "\n";
	print_type<value_type>::print(file);

	// Write element matrices.
	for(element_iterator it = element_begin(); it != element_end(); ++it) {
	    // Write indices.
	    for(int i = 0; i < it->nb_vars()-1; ++i) {
		file << it->get_indices()(i) << " ";
	    }
	    file << it->get_indices()(it->nb_vars()-1) << "\n";

	    // Write values.
	    for(int r = 0; r < it->nb_vars(); ++r) {
		for(int c = 0; c < it->nb_vars()-1; ++c) {
		    print_value<value_type>::print(file, it->get_values()(r,c));
		}
		print_value<value_type>::print(
					       file, it->get_values()(r,it->nb_vars()-1)
					       );
		file << "\n";
	    }
	    file << "\n";
	}
    }
    
        

    /*******************************************************************************
     * Data Members
     ******************************************************************************/
//   private:
  public:
    /**
     * The total number of elements.
     */
    int m_total_elements;

    /**
     * The total number of variables.
     */
    int m_total_vars;

    /**
     * The elements of the grid, stored consecutively.
     */
    element_type* m_elements;
};

}

#endif // MTL_ELEMENT_STRUCTURE_INCLUDE
