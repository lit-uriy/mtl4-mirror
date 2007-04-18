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
      : AlgebraicCollection<T>
    {
	const_reference T::operator() (size_type row, size_type col) const;

	size_type nnz(T);

	// A[r][c] equivalent to A(r, c)
    };
#else
    /// Concept Matrix
    /**
        \par Refinement of:
	- AlgebraicCollection < T >
	\par Notation:
	- X is a type that models Matrix
	- A is an object of type X
	- r, c are objects of size_type
	\par Valid expressions:
	- Element access: \n A(r, c) \n Return Type: const_reference 
	  \n Semantics: Element in row \p r and column \p c
	- Element access: \n A[r][c] \n Equivalent to A(r, c)
	\par Models:
	- dense2D
	- morton_dense
	- compressed2D
	\note
	-# The access via A[r][c] is supposed to be implemented by means of A(r, c) (typically via CRTP and proxies).
	  If it would become (extremely) important to support 2D C arrays, it might be necessary to drop the requirement
	  of element access by A(r, c).
	-# The name const_reference does not imply that the return type is necessarily referrable. For instance compressed2D
	   returns value_type.
     */ 
    template <typename T>
    struct Matrix
	: public AlgebraicCollection<T>
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
	\par Models:
	- mtl::matrix::inserter < T >
	\note
	-# Used in concept InsertableMatrix
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
	\par Models:
	- dense2D
	- morton_dense
	- compressed2D
	\note
	-# All matrices in MTL model this concept in order and all future matrices are supposed to.
    */
    template <typename T>
    struct InsertableMatrix
      : Matrix < T >
    {};
#endif


#ifdef __GXX_CONCEPTS__
    concept MutableMatrix<typename T>
      : Matrix<T>,
	MutableCollection<T>
    {
	reference T::operator() (size_type row, size_type col);

	// A[r][c] equivalent to A(r, c)
    };
#else
    /// Concept MutableMatrix
    /**
        \par Refinement of:
	- Matrix < T >
	- MutableCollection < T >
	\par Notation:
	- X is a type that models MutableMatrix
	- A is an object of type X
	- r, c are objects of size_type
	\par Valid expressions:
	- Element access: \n A(r, c) \n Return Type: reference 
	  \n Semantics: Element in row \p r and column \p c
	- Element access: \n A[r][c] \n Equivalent to A(r, c)
	\par Models:
	- dense2D
	- morton_dense
	\note
	-# The access via A[r][c] is supposed to be implemented by means of A(r, c) (typically via CRTP and proxies).
	  If it would become (extremely) important to support 2D C arrays, it might be necessary to drop the requirement
	  of element access by A(r, c).
     */ 
    template <typename T>
    struct MutableMatrix
	: public Matrix<T>,
	  public MutableCollection<T>
    {
	/// Element access (in addition to const access)
	reference T::operator() (size_type row, size_type col);
    };
#endif

    
#ifdef __GXX_CONCEPTS__
    concept ConstantSizeMatrix<typename T>
      : Matrix<T>,
	ConstantSizeAlgebraicCollection<T>
    {};
#else
    /// Concept ConstantSizeMatrix
    /**
        \par Refinement of:
	- Matrix < T >
	- ConstantSizeAlgebraicCollection < T >
     */ 
    template <typename T>
    struct ConstantSizeMatrix
      : public Matrix<T>,
	public ConstantSizeAlgebraicCollection<T>
    {};
#endif

    

/*@}*/ // end of group Concepts

} // namespace mtl

#endif // MTL_MATRIX_CONCEPT_INCLUDE
