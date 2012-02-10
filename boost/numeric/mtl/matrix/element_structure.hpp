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


#include <boost/numeric/mtl/matrix/element.hpp>
// #include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/mtl/utility/ashape.hpp>

namespace mtl { namespace matrix {

#if 0
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
  #endif

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
    /// Type of the numerical values of the element coefficient matrices.
    typedef ValueType value_type;

    /// Type of the element.
    typedef element<value_type> element_type;

    /// Type of index arrays.
    typedef typename element_type::index_type index_type;

    /// Type of the iterator over the elements of the mesh.
    typedef element_type* element_iterator;

    /// Type of this class.
    typedef element_structure<ValueType> this_type;
    typedef this_type                    self;
    

    /*******************************************************************************
     * Constructors
     ******************************************************************************/


    /// Standard constructor.
  public:
    element_structure(int total_elements, int total_vars, element_type* elements)
      : m_total_elements(total_elements),
	m_total_vars(total_vars),
	m_elements(elements)
    { }

    /// Copy the given mesh.
  public:
    element_structure(this_type const& other)
      : m_total_elements(other.m_total_elements),
	m_total_vars(other.m_total_vars),
	m_elements(m_total_elements == 0 ? 0 : new element_type[m_total_elements])
    {
	typedef typename element_type::neighbour_collection_type neigh_coll_type;

	int j = 0;
	bool ordered = true;
	for(element_iterator it = other.element_begin(); it != other.element_end(); ++it) {
	    // Deep copy the elements.
	    m_elements[j] = *it;
	    ordered &= (it->get_id() == j);
	    ++j;
	}
	assert( ordered );
	// Reconstruct the network of neighbours.
	for(element_iterator it = this->element_begin(); it != this->element_end(); ++it) {
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


    ~element_structure() { delete[] m_elements; }

    /*******************************************************************************
     * Inspector Members
     ******************************************************************************/

    /// Total number of elements in the grid.
    int get_total_elements() const { return m_total_elements; }

    /// Total number of variables.
    int get_total_vars() const { return m_total_vars;   }

    /// Total number of non-zero values.
    int get_total_nnz() const 
    {
	int nnz = 0;
	for(element_iterator it = element_begin(); it != element_end(); ++it) {
	    nnz += it->nnz();
	}
	return nnz;
    }

    /// Iterator to the first element.
    element_iterator element_begin() const { return m_elements + 0;  }

    /// An iterator to the element past the last element.
    element_iterator element_end() const { return m_elements + this->get_total_elements();   }

    /*******************************************************************************
     * File Operations
     ******************************************************************************/
  public:

    /**
     * Writes the elements to the specified file.
     *
     * TODO test this code
     */
#if 0
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
		print_value<value_type>::print(file, it->get_values()(r,it->nb_vars()-1));
		file << "\n";
	    }
	    file << "\n";
	}
    }
#endif
        

    /*******************************************************************************
     * Data Members
     ******************************************************************************/
//   private:
  public:
    int m_total_elements; ///< The total number of elements.
    int m_total_vars; ///< The total number of variables.
    element_type* m_elements; ///< The elements of the grid, stored consecutively.
};

template <typename ValueType>
inline std::size_t num_rows(const element_structure<ValueType>& A)
{   return A.get_total_vars(); }

template <typename ValueType>
inline std::size_t num_cols(const element_structure<ValueType>& A)
{   return A.get_total_vars(); }

template <typename ValueType>
inline std::size_t size(const element_structure<ValueType>& A)
{   return A.get_total_vars() * A.get_total_vars(); }


template <typename ValueType>
inline void swap(element_structure<ValueType>& x, element_structure<ValueType>& y)
{
    swap(x.m_total_elements, y.m_total_elements);
    swap(x.m_total_vars, y.m_total_vars);
    swap(x.m_elements, y.m_elements);
}


}} // mtl::matrix


#endif // MTL_ELEMENT_STRUCTURE_INCLUDE
