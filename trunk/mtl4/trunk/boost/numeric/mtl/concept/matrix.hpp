// $COPYRIGHT$

#ifndef MTL_MATRIX_CONCEPT_INCLUDE
#define MTL_MATRIX_CONCEPT_INCLUDE

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>

#ifdef __GXX_CONCEPTS__
#  include <concepts>
#else 
#  include <boost/numeric/linear_algebra/pseudo_concept.hpp>
#endif

namespace mtl {

/** @addtogroup Concepts
 *  @{
 */

#ifdef __GXX_CONCEPTS__
    concept Matrix<typename T>
      : Collection<T>
    {
	const_reference T::operator() (size_type row, size_type col) const;

	size_type num_rows(T);
	size_type num_cols(T);
	size_type nnz(T);

	// A[r][c] equivalent to A(r, c)
    };
#else
    /// Concept Matrix
    /**
        \par Refinement of:
	- Collection < T >
	\par Notation:
	- X is a type that models Matrix
	- A is an object of type X
	- r, c are objects of size_type
	\par Valid expressions:
	- Element access: \n A(r, c) \n Return Type: const_reference 
	  \n Semantics: Element in row \p r and column \p c
	- Element access: \n A[r][c] \n Equivalent to A(r, c)
	- Number of rows: \n num_rows(A) \n Return Type: size_type
	- Number of columns: \n num_cols(A) \n Return Type: size_type
	\invariant
     */ 
    template <typename T>
    struct Matrix
	: Collection<T>
    {
	/// Element access
	const_reference T::operator() (size_type row, size_type col) const;
    };
#endif

    

#ifdef __GXX_CONCEPTS__
    concept MatrixInserter<typename T>
    {
	typename matrix_type;
	// typename T::matrix_type;

	requires Matrix<matrix_type>;
	
	typename proxy_type;
	proxy_type operator() (Matrix<matrix_type>::size_type row, Matrix<matrix_type>::size_type col);
	
	T operator<< (proxy_type, Matrix<matrix_type>::value_type>);
    };
#else
    /// Concept MatrixInserter: classes that enable efficient insertion into matrices, esp. compressed sparse.
    /** 
	Used to fill non-mutable matrices like compressed2D. Matrix inserters might be parametrizable with
	update functor. This allow to perform different operations when entry already exist, e.g. overwriting,
	incrementing, minimum, ... The most important updates are certainly overwrite and increment (add).

	\par Associated types
	- matrix_type

	\par Requires:
	- Matrix<matrix_type>
	
	\par Notation:
	- X is a type that models MatrixInserter
	- A is an object of type X
	- r, c are objects of type Matrix<matrix_type>::size_type
	- v is an object of type Matrix<matrix_type>::value_type

	\par Valid expressions:
	- Insertion with shift operator: \n
	   A(r, c) << v \n
	   Return type: T
	\notes
	- Used in concept InsertableMatrix
	\par Models:
	- mtl::matrix::inserter < T >
     */
    template <typename T>
    struct MatrixInserter
    {
	/// Type  of matrix into which is inserted
	typedef associated_type matrix_type;

	/// Return type of element access; only proxy
	typedef associated_type  proxy_type;
	/// Element access; returns a proxy that handles insertion
	proxy_type operator() (Matrix<matrix_type>::size_type row, Matrix<matrix_type>::size_type col);
    };
#endif

#ifdef __GXX_CONCEPTS__
    concept InsertableMatrix<typename T>
      : Matrix<T>
    {
	requires MatrixInserter<mtl::matrix::inserter<T> >;
    };
#else
    /// Concept InsertableMatrix: %matrix that can be filled by means of inserter
    /** 
	\par Requires:
	- MatrixInserter < mtl::matrix::inserter< T > >
    */
    template <typename T>
    struct InsertableMatrix
      : Matrix < T >
    {};
#endif


/*@}*/ // end of group Concepts

} // namespace mtl

#endif // MTL_MATRIX_CONCEPT_INCLUDE
