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

/*@}*/ // end of group Concepts

} // namespace mtl

#endif // MTL_MATRIX_CONCEPT_INCLUDE
