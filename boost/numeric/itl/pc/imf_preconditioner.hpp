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
 * The IMF preconditioner.
 *
 * References
 * 	[1] N. Vannieuwenhoven and K. Meerbergen, IMF: An incomplete multifron-
 *		tal LU-factorization for element-structured sparse linear systems,
 *		Tech. Rep. TW581, Department of Computer Science, KULeuven,
 *		December 2010.
 */

#ifndef MTL_IMF_PRECONDITIONER_INCLUDE
#define MTL_IMF_PRECONDITIONER_INCLUDE

#include <boost/numeric/itl/pc/matrix_algorithms.hpp>
#include <boost/numeric/mtl/matrix/compressed2D.hpp>
#include <boost/numeric/mtl/matrix/coordinate2D.hpp>
#include <boost/numeric/mtl/matrix/dense2D.hpp>
#include <boost/numeric/mtl/vector/dense_vector.hpp>



namespace itl {   namespace pc {



/**
 * The IMF preconditioner, as described in [1].
 */
template<class ValType>
class imf_preconditioner {

/*******************************************************************************
 * Type Definitions
 ******************************************************************************/

public:
	/**
	 * The type of the values.
	 */
	typedef ValType value_type;

	/**
	 * The type of the sparse data structure for the upper matrices.
	 */
	typedef mtl::matrix::coordinate2D<value_type> sparse_type_upper;

	/**
	 * The type of the sparse data structure for the lower matrices.
	 */
	typedef mtl::matrix::coordinate2D<value_type> sparse_type_lower;

	/**
	 * The type of the vectors.
	 */
	typedef mtl::dense_vector<value_type> vector_type;

	/**
	 * The type of the permutation vector.
	 */
	typedef mtl::dense_vector<int>  index_type;

	/**
	 * The type of the matrices on the block diagonal.
	 */
	typedef mtl::dense2D<value_type>  block_type;

	/**
	 * The type of the sequence of lower matrices.
	 */
	typedef std::vector<sparse_type_lower*> lower_matrix_coll_type;

	/**
	 * The type of the sequence of upper matrices.
	 */
	typedef std::vector<sparse_type_upper*> upper_matrix_coll_type;



	typedef mtl::dense2D<value_type>    blas_matrix;

/*******************************************************************************
 * Constructors
 ******************************************************************************/

public:
	template< class ElementStructure >
	imf_preconditioner(
		const ElementStructure& element_structure ,
  		const int maxlofi=0
	)
	: 	m_nb_vars( element_structure.get_total_vars() ),
	  	m_ordering( element_structure.get_total_vars() ),
	  	m_diagonal_index(0),
	  	m_diagonal(0)
	{
		factor(element_structure, maxlofi);
	}

	/**
	 * Disallow the copy constructor and assignment.
	 */
private:
	imf_preconditioner();
	imf_preconditioner(const imf_preconditioner& other);
	void operator=(const imf_preconditioner& other);

	/**
	 * Constructs the IMF preconditioner.
	 */
private:
	template< class ElementStructure >
  	void factor(const ElementStructure&, const int); 


/*******************************************************************************
 * Destructor
 ******************************************************************************/

public:
	~imf_preconditioner() {
		if(m_diagonal_index) {
			delete[] m_diagonal_index;
		}
		m_diagonal_index = 0;
		if(m_diagonal) {
			delete[] m_diagonal;
		}
		m_diagonal = 0;
	}

/*******************************************************************************
 * Inspectors
 ******************************************************************************/

public:

	/**
	 * Returns the number of levels (equals the number of lower and upper
	 * matrices).
	 */
	int get_nb_levels() const {
		return m_levels;
	}

	/**
	 * Returns the number of blocks on the diagonal.
	 */
	int get_nb_blocks() const {
		return m_nb_blocks;
	}

/*******************************************************************************
 * Preconditioning
 ******************************************************************************/

	/**
	 * Applies the preconditioner to the given matrix.
	 */
public:
	//equals operator() in Nicks code
	template <typename Vector>
	Vector solve(const Vector& b) const {
	    mtl::matrix::traits::permutation<>::type P(permutation(m_ordering));//TODO change in loop
	    Vector m_tmp(b), m(trans(P)*b);
	    m_tmp = imf_apply(m);
	    for(unsigned int i=0; i< size(b); i++)
		  m[i]= m_tmp[m_ordering[i]];
	    return m; 
	}  

	/**
	 * Applies the preconditioner.
	 */
private:  //prototypes
	template< class Vector >
	Vector imf_apply(const Vector&) const;
/*******************************************************************************
 * Data Members
 ******************************************************************************/

private:
	/**
	 * The number of variables (also the size of the m_ordering vector).
	 */
	unsigned int m_nb_vars;

	/**
	 * The number of blocks on the diagonal.
	 */
	unsigned int m_nb_blocks;

	/**
	 * A vector containing the renumbering of IMF.
	 */
	index_type m_ordering;

	/**
	 * The number of levels (equals the number of entries in the diagonal index
	 * array minus one).
	 */
	int m_levels;

	/**
	 * The index array for the matrices on the block diagonal. The i^th entry
	 * indicates where the i^th level of block diagonal matrices starts in the
	 * right hand side vector.
	 */
	int* m_diagonal_index;

	/**
	 * The matrices on the block diagonal.
	 */
	block_type* m_diagonal;

	/**
	 * The sparse lower matrices of each level, sorted by level.
	 */
	lower_matrix_coll_type m_lower;

	/**
	 * The sparse upper matrices of each level, sorted by level.
	 */
	upper_matrix_coll_type m_upper;
};

/// Solve 
template <typename Matrix, typename Vector>
Vector solve(const imf_preconditioner<Matrix>& P, const Vector& b)
{
	return P.solve(b);
}
}//namespace pc
}//namespace itl


#endif // MTL_IMF_PRECONDITIONER_INCLUDE
