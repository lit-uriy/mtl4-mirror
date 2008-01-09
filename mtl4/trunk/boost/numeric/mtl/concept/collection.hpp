// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

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
    concept AlgebraicCollection<typename T>
      : Collection<T>
    {
	size_type num_rows(T);
	size_type num_cols(T);
	size_type size(T);
    };
#else
    /// Concept AlgebraicCollection: common requirements of matrices, vectors, and scalars in computations
    /** For more design clarity we consider them all as matrices (as Matlab does) and we regard 
	Scalar and Vector as special cases (see there).  However, the implementation of vectors
	is supposed to differ from the ones of matrices in order to provide efficient computations and storage.
        \par Refinement of:
	- Collection < T >
	\par Notation:
	- X is a type that models AlgebraicCollection
	- x is an object of type X
	\par Valid expressions:
	- Number of rows: \n num_rows(x) \n Return Type: size_type
	- Number of columns: \n num_cols(x) \n Return Type: size_type
	- Number of elements: \n size(x) \n Return Type: size_type
	  \n Sematics: num_rows(x) * num_cols(x) (but possibly faster implemented)
    */
    template <typename T>
    struct AlgebraicCollection
	: public Collection<T>
    {};
#endif


#ifdef __GXX_CONCEPTS__
    concept ConstantSizeAlgebraicCollection<typename T>
      : AlgebraicCollection<T>,
        ConstantSizeCollection<T>
    {
	typename static_num_rows;
	typename static_num_cols;
	typename static_size;
    };
#else
    /// Concept ConstantSizeAlgebraicCollection: extension of AlgebraicCollection with meta-functions
    /** This concept is used for algebraic collections with sizes known at compile time. 
	The motivation is that if the size of the collection is
	is small, arithmetic operations can be unrolled at compile time.

        \par Refinement of:
	- Collection < T >
	\par Notation:
	- X is a type that models ConstantSizeAlgebraicCollection
	- x is an object of type X
	\par Valid expressions:
	- Number of rows: \n static_num_rows<X>::value
	- Number of columns: \n static_num_cols<X>::value
	- Number of elements: \n static_size<X>::value
	  \n Sematics: static_num_rows<X>::value * static_size<X>::value
	\note
	-# For more design clarity we consider them all as matrices (as Matlab does) and we regard 
	   Scalar and Vector as special cases (see there).  However, the implementation of vectors
	   is supposed to differ from the ones of matrices in order to provide efficient computations and storage.

    */
    template <typename T>
    struct ConstantSizeAlgebraicCollection
      : public AlgebraicCollection<T>,
        public ConstantSizeCollection<T>
    {
	/// Associated type: meta-function for number of rows
	typedef associated_type static_num_rows;
	/// Associated type: meta-function for number of columns
	typedef associated_type static_num_cols;
	/// Associated type: meta-function for number of elements
	typedef associated_type static_size;
    };
#endif



#ifdef __GXX_CONCEPTS__
    concept TraversableCollection<typename Tag, typename C>
      : Collection<C>
    {
	typename cursor_type;

	cursor_type begin<Tag>(const C& c);
	cursor_type   end<Tag>(const C& c);
    }
#else
    /// Concept TraversableCollection: collections that can be traversed by cursor or iterator
    template <typename Tag, typename C>
    struct TraversableCollection
	: public Collection<C>
    {
	/// Associated type: return type of tagged begin and end function
	typedef associated_type cursor_type;

	/// Tagged free function that returns a cursor or iterator at the begin of an interval 
	/** The interval is specified by the Tag, i.e. the function is called begin<Tag>(c); */
	cursor_type begin(const C& c);

	/// Tagged free function that returns a cursor or iterator at the end of an interval 
	/** The interval is specified by the Tag, i.e. the function is called end<Tag>(c);  */
	cursor_type end(const C& c);
    };
#endif


#ifdef __GXX_CONCEPTS__
    concept TraversableMutableCollection<typename Tag, typename C>
      : MutableCollection<C>
    {
	typename cursor_type;

	cursor_type begin<Tag>(C& c);
	cursor_type   end<Tag>(C& c);
    }
#else
    /// Concept TraversableMutableCollection: collections that can be traversed by (mutable) iterator
    template <typename Tag, typename C>
    struct TraversableMutableCollection
	: public MutableCollection<C>
    {
	/// Associated type: return type of tagged begin function
	typedef associated_type cursor_type;

	/// Tagged free function that returns a cursor or iterator at the begin of an interval 
	/** The interval is specified by the Tag, i.e. the function is called begin<Tag>(c); */
	cursor_type begin(const C& c);

	/// Tagged free function that returns a cursor or iterator at the end of an interval 
	/** The interval is specified by the Tag, i.e. the function is called end<Tag>(c);  */
	cursor_type end(const C& c);
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


#ifdef __GXX_CONCEPTS__
    concept OrientedCollection<typename T>
      : Collection<T>
    {
	typename orientation;

    };
#else
    /// Concept OrientedCollection: collections with concept-awareness in terms of associated type
    /** Concept-awareness is given for matrices as well as for vectors consistent to the unification in
	AlgebraicCollection. The orientation of vectors determines whether it is a row or a column vector.
	The orientation of matrices only characterizes the internal representation and has no semantic consequences.
        \par Refinement of:
	- Collection < T >
	\par Associated type:
	- orientation
    */
    template <typename T>
    struct OrientedCollection
	: public Collection<T>
    {
	/// Associated type for orientation; by default identical with member type
	typedef typename T::orientation orientation;
    };
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
        // Alleged ambiguity with mtl::tag::dense2D on MSVC
        typedef typename mtl::dense2D<Value, Parameters>::size_type size_type;
    };
#endif


#ifdef __GXX_CONCEPTS__
    template <typename Value, unsigned long Mask, typename Parameters>
    concept_map Collection<morton_dense<Value, Mask, Parameters> >
    {
	typedef Value            value_type;
	typedef const Value&     const_reference;
	typedef typename morton_dense<Value, Mask, Parameters>::size_type size_type;
    };
#else
    template <typename Value, unsigned long Mask, typename Parameters>
    struct Collection<morton_dense<Value, Mask, Parameters> >
    {
	typedef Value            value_type;
	typedef const Value&     const_reference;
	typedef typename morton_dense<Value, Mask, Parameters>::size_type size_type;
    };
#endif


#ifdef __GXX_CONCEPTS__
    template <typename Value, typename Parameters>
    concept_map Collection<compressed2D<Value, Parameters> >
    {
	typedef Value            value_type;
	typedef const Value&     const_reference;
	typedef typename compressed2D<Value, Parameters>::size_type size_type;
    };
#else
    template <typename Value, typename Parameters>
    struct Collection<compressed2D<Value, Parameters> >
    {
	typedef Value            value_type;
	typedef Value            const_reference;
	typedef typename compressed2D<Value, Parameters>::size_type size_type;
    };

#endif


#ifdef __GXX_CONCEPTS__
    template <typename Scaling, typename Coll>
    concept_map Collection<matrix::scaled_view<Scaling, Coll> >
    {
	typedef typename matrix::scaled_view<Scaling, Coll>::value_type        value_type;
	typedef typename matrix::scaled_view<Scaling, Coll>::const_reference   const_reference;
	typedef typename matrix::scaled_view<Scaling, Coll>::size_type         size_type;
    };
#else
    template <typename Scaling, typename Coll>
    struct Collection<matrix::scaled_view<Scaling, Coll> >
    {
	typedef typename matrix::scaled_view<Scaling, Coll>::value_type        value_type;
	typedef typename matrix::scaled_view<Scaling, Coll>::const_reference   const_reference;
	typedef typename matrix::scaled_view<Scaling, Coll>::size_type         size_type;
    };
#endif


#ifdef __GXX_CONCEPTS__
    template <typename Scaling, typename Coll>
    concept_map Collection<vector::scaled_view<Scaling, Coll> >
    {
	typedef typename vector::scaled_view<Scaling, Coll>::value_type        value_type;
	typedef typename vector::scaled_view<Scaling, Coll>::const_reference   const_reference;
	typedef typename vector::scaled_view<Scaling, Coll>::size_type         size_type;
    };
#else
    template <typename Scaling, typename Coll>
    struct Collection<vector::scaled_view<Scaling, Coll> >
    {
	typedef typename vector::scaled_view<Scaling, Coll>::value_type        value_type;
	typedef typename vector::scaled_view<Scaling, Coll>::const_reference   const_reference;
	typedef typename vector::scaled_view<Scaling, Coll>::size_type         size_type;
    };
#endif


#ifdef __GXX_CONCEPTS__
    template <typename Coll>
    concept_map Collection<matrix::conj_view<Coll> >
    {
	typedef typename matrix::conj_view<Coll>::value_type        value_type;
	typedef typename matrix::conj_view<Coll>::const_reference   const_reference;
	typedef typename matrix::conj_view<Coll>::size_type         size_type;
    };
#else
    template <typename Coll>
    struct Collection<matrix::conj_view<Coll> >
    {
	typedef typename matrix::conj_view<Coll>::value_type        value_type;
	typedef typename matrix::conj_view<Coll>::const_reference   const_reference;
	typedef typename matrix::conj_view<Coll>::size_type         size_type;
    };
#endif


#ifdef __GXX_CONCEPTS__
    template <typename Coll>
    concept_map Collection<vector::conj_view<Coll> >
    {
	typedef typename vector::conj_view<Coll>::value_type        value_type;
	typedef typename vector::conj_view<Coll>::const_reference   const_reference;
	typedef typename vector::conj_view<Coll>::size_type         size_type;
    };
#else
    template <typename Coll>
    struct Collection<vector::conj_view<Coll> >
    {
	typedef typename vector::conj_view<Coll>::value_type        value_type;
	typedef typename vector::conj_view<Coll>::const_reference   const_reference;
	typedef typename vector::conj_view<Coll>::size_type         size_type;
    };
#endif


#ifdef __GXX_CONCEPTS__
    template <typename Matrix>
    concept_map Collection<matrix::hermitian_view<Matrix> >
    {
	typedef typename matrix::hermitian_view<Matrix>::value_type        value_type;
	typedef typename matrix::hermitian_view<Matrix>::const_reference   const_reference;
	typedef typename matrix::hermitian_view<Matrix>::size_type         size_type;
    };
#else
    template <typename Matrix>
    struct Collection<matrix::hermitian_view<Matrix> >
    {
	typedef typename matrix::hermitian_view<Matrix>::value_type        value_type;
	typedef typename matrix::hermitian_view<Matrix>::const_reference   const_reference;
	typedef typename matrix::hermitian_view<Matrix>::size_type         size_type;
    };
#endif


#ifdef __GXX_CONCEPTS__
    template <typename Value, typename Parameters>
    concept_map Collection<vector::dense_vector<Value, Parameters> >
    {
	typedef Value            value_type;
	typedef const Value&     const_reference;
	typedef typename vector::dense_vector<Value, Parameters>::size_type size_type;
    };
#else
    template <typename Value, typename Parameters>
    struct Collection<vector::dense_vector<Value, Parameters> >
    {
	typedef Value            value_type;
	typedef const Value&     const_reference;
	typedef typename vector::dense_vector<Value, Parameters>::size_type size_type;
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

#ifdef __GXX_CONCEPTS__

    template <typename Value, typename Parameters>
    concept_map MutableCollection<morton_dense<Value, Parameters> >
    {
	typedef Value&           reference;
    };

#else

    template <typename Value, unsigned long Mask, typename Parameters>
    struct MutableCollection<morton_dense<Value, Mask, Parameters> >
	: public Collection<morton_dense<Value, Mask, Parameters> >
    {
	typedef Value&           reference;
    };

#endif

#ifdef __GXX_CONCEPTS__
    template <typename Scaling, typename Coll>
    concept_map OrientedCollection< matrix::scaled_view<Scaling, Coll> >
    {
	typedef typename OrientedCollection<Coll>::orientation       orientation;
    };
#else
    template <typename Scaling, typename Coll>
    struct OrientedCollection< matrix::scaled_view<Scaling, Coll> >
    {
	typedef typename OrientedCollection<Coll>::orientation       orientation;
    };
#endif


#ifdef __GXX_CONCEPTS__
    template <typename Scaling, typename Coll>
    concept_map OrientedCollection< vector::scaled_view<Scaling, Coll> >
    {
	typedef typename OrientedCollection<Coll>::orientation       orientation;
    };
#else
    template <typename Scaling, typename Coll>
    struct OrientedCollection< vector::scaled_view<Scaling, Coll> >
    {
	typedef typename OrientedCollection<Coll>::orientation       orientation;
    };
#endif


#ifdef __GXX_CONCEPTS__
    template <typename Coll>
    concept_map OrientedCollection<matrix::conj_view<Coll> >
    {
	typedef typename OrientedCollection<Coll>::orientation       orientation;
    };
#else
    template <typename Coll>
    struct OrientedCollection<matrix::conj_view<Coll> >
    {
	typedef typename OrientedCollection<Coll>::orientation       orientation;
    };
#endif


#ifdef __GXX_CONCEPTS__
    template <typename Coll>
    concept_map OrientedCollection<vector::conj_view<Coll> >
    {
	typedef typename OrientedCollection<Coll>::orientation       orientation;
    };
#else
    template <typename Coll>
    struct OrientedCollection<vector::conj_view<Coll> >
    {
	typedef typename OrientedCollection<Coll>::orientation       orientation;
    };
#endif


#ifdef __GXX_CONCEPTS__
    template <typename Functor, typename Coll>
    concept_map OrientedCollection< vector::map_view<Functor, Coll> >
    {
	typedef typename OrientedCollection<Coll>::orientation       orientation;
    };
#else
    template <typename Functor, typename Coll>
    struct OrientedCollection< vector::map_view<Functor, Coll> >
    {
	typedef typename OrientedCollection<Coll>::orientation       orientation;
    };
#endif


#ifdef __GXX_CONCEPTS__
    template <typename Coll>
    concept_map OrientedCollection<matrix::hermitian_view<Coll> >
    {
	typedef typename transposed_orientation<typename OrientedCollection<Coll>::orientation>::type   orientation;
    };
#else
    template <typename Coll>
    struct OrientedCollection<matrix::hermitian_view<Coll> >
    {
	typedef typename transposed_orientation<typename OrientedCollection<Coll>::orientation>::type   orientation;
    };
#endif



/*@}*/ // end of group Concepts

} // namespace mtl

#endif // MTL_COLLECTION_INCLUDE
