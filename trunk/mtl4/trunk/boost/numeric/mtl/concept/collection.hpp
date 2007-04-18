// $COPYRIGHT$

#ifndef MTL_COLLECTION_INCLUDE
#define MTL_COLLECTION_INCLUDE

#include <boost/numeric/mtl/mtl_fwd.hpp>

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
    concept Collection<typename T>
    {
	typename value_type;
	typename const_reference;
	typename size_type;
    };
#else
    /// Concept Collection
    template <typename T>
    struct Collection
    {
	/// Associated type: elements in the collection
	typedef associated_type value_type;

	/// Associated type: return type of const element access (however implemented)
	typedef associated_type const_reference;

	/// Associated type: size type used for indexing in collection
	typedef associated_type size_type;
    };
#endif


#ifdef __GXX_CONCEPTS__
    concept MutableCollection<typename T>
      : Collection<T>
    {
	typename reference;
    }
#else
    /// Concept MutableCollection
    template <typename T>
    struct MutableCollection
	: public Collection<T>
    {
	/// Associated type: return type of non-const element access (however implemented)
	typedef associated_type reference;
    };
#endif


#ifdef __GXX_CONCEPTS__
    concept ConstantSizeCollection<typename T>
      : Collection<T>
    {};
#else
    /// Concept ConstantSizeCollection: size parameters of collection are completely given at compile time
    /* Which parameters determine collection size depends on type of collection, e.g. different for vector and matrix
       \par Refinement of:
       - Collection < T >
    */
    template <typename T>
    struct ConstantSizeCollection
	: Collection<T>
    {};
#endif


#ifdef __GXX_CONCEPTS__
    concept AlgebraCollection<typename T>
      : Collection<T>
    {
	size_type num_rows(T);
	size_type num_cols(T);
	size_type size(T);
    };
#else
    /// Concept AlgebraCollection: common requirements of matrices, vectors, and scalars in computations
    /** For more design clarity we consider them all as matrices (as Matlab does) and we regard 
	Scalar and Vector as special cases (see there).  However, the implementation of vectors
	is supposed to differ from the ones of matrices in order to provide efficient computations and storage.
        \par Refinement of:
	- Collection < T >
	\par Notation:
	- X is a type that models Matrix
	- A is an object of type X
	\par Valid expressions:
	- Number of rows: \n num_rows(A) \n Return Type: size_type
	- Number of columns: \n num_cols(A) \n Return Type: size_type
	- Number of elements: \n size(A) \n Return Type: size_type
	  \n Sematics: num_rows(A) * num_cols(A) (but possibly faster implemented)
    */
    template <typename T>
    struct AlgebraCollection
	: Collection<T>
    {
    };
#endif







#ifdef __GXX_CONCEPTS__
#if 0
    concept CategorizedType<typename T>
    {
	typedef associated_type type;
    };
#endif
#endif







// ============================================
// Concept maps (and emulations as type traits)
// ============================================

#ifdef __GXX_CONCEPTS__

    template <typename Value, typename Parameters>
    concept_map Collection<dense2D<Value, Parameters> >
    {
	typedef Value            value_type;
	typedef const Value&     const_reference;
	typedef typename dense2D<Value, Parameters>::size_type size_type;
    };

#else

    template <typename Value, typename Parameters>
    struct Collection<dense2D<Value, Parameters> >
    {
	typedef Value            value_type;
	typedef const Value&     const_reference;
	typedef typename dense2D<Value, Parameters>::size_type size_type;
    };

#endif


#ifdef __GXX_CONCEPTS__

    template <typename Value, typename Parameters>
    concept_map MutableCollection<dense2D<Value, Parameters> >
    {
	typedef Value&           reference;
    };

#else

    template <typename Value, typename Parameters>
    struct MutableCollection<dense2D<Value, Parameters> >
	: public Collection<dense2D<Value, Parameters> >
    {
	typedef Value&           reference;
    };

#endif

/*@}*/ // end of group Concepts

} // namespace mtl

#endif // MTL_COLLECTION_INCLUDE
