// $COPYRIGHT$

#ifndef MTL_TAG_INCLUDE
#define MTL_TAG_INCLUDE

#include <boost/numeric/mtl/utility/glas_tag.hpp>

namespace mtl { namespace tag {


// For internal use (e.g., to invalidate range generators)
// Is this the same as bottom?
struct unsupported {};

/// Tag for all types
struct universe {};

// Tag for any scalar value
/** At the moment default for unknown types (will be precised later) */
struct scalar : virtual universe {};

/// For non-MTL types with category not explicitly defined
/** At the moment all treated as scalars (will be precised later) */
struct unknown : virtual scalar {};

/// Tag for any MTL vector (and user-declared MTL vectors)
struct vector : virtual universe {};

/// Tag for any MTL column vector (and user-declared MTL vectors)
struct col_vector : virtual vector {};

/// Tag for any MTL row vector (and user-declared MTL vectors)
struct row_vector : virtual vector {};

/// Tag for any MTL matrix (and user-declared MTL matrices)
struct matrix : virtual universe {};

/// Tag for any dense collection
struct dense : virtual universe {};
    
/// Tag for vectors with one-dimensional memory addressing
/** offet v_i is x*i for some x */
struct has_1D_layout : virtual dense {};
    
/// Tag for matrices with two-dimensional memory addressing
/** offet a_ij is x*i + y*j for some x and y */
struct has_2D_layout : virtual dense {};
    
/// Tag for any sparse collection
struct sparse : virtual universe {};
    
/// Tag for collections where values are stored contigously in memory
struct contiguous_memory : virtual universe {};

/// Tag for dense and contiguous collections
/** Only short cut */
struct contiguous_dense : virtual dense, virtual contiguous_memory {};

/// Collection with iterator
struct has_iterator : virtual universe {};

/// Collection with random-access iterator
struct has_ra_iterator : virtual has_iterator {};

/// Collection with fast random-access iterator
/** Meaning: unrolling is probably beneficial. Counter-example: Morton-ordered matrices have
    random access but this is so slow that regular traversal is favorable */
struct has_fast_ra_iterator : virtual has_ra_iterator {};

/// Collection with cursor
struct has_cursor : virtual universe {};

/// Collection with random-access cursor
struct has_ra_cursor : virtual has_cursor {};

/// Collection with fast random-access cursor
/** Meaning: unrolling is probably beneficial. Counter-example: Morton-ordered matrices have
    random access but this is so slow that regular traversal is favorable */
struct has_fast_ra_cursor : virtual has_ra_cursor {};

/// Tag for matrices with sub_matrix function exist and doesn't say for which ranges it is defined
struct has_sub_matrix : virtual universe {};

/// Sub-dividable into quadrants, i.e. arbitrary sub-matrices not necessarily supported but recursion works
//  more explanation needed
struct qsub_dividable : virtual has_sub_matrix {};

/// Tag for sub-dividable matrix, i.e. sub_matrix works 
struct sub_dividable : virtual qsub_dividable {};

/// Tag for dense row vector in the category lattice
struct dense_row_vector
  : virtual row_vector, virtual contiguous_dense, 
    virtual has_1D_layout
{};

/// Tag for dense column vector in the category lattice
struct dense_col_vector
  : virtual col_vector, virtual contiguous_dense, 
    virtual has_1D_layout
{};

/// Tag for a map of a (regular) dense matrix in the category lattice
/** The map perform address computation and has therefore no 2D-layout */
struct dense2D_map 
  : virtual matrix, virtual contiguous_dense, virtual has_fast_ra_cursor, 
    virtual has_fast_ra_iterator, virtual sub_dividable
{};

/// Tag for (regular) dense matrix in the category lattice
struct dense2D 
  : virtual dense2D_map, virtual has_2D_layout
{};

/// Tag for Morton-order matrix in the category lattice
struct morton_dense 
  : virtual matrix, virtual contiguous_dense, virtual has_ra_cursor, 
    virtual has_ra_iterator, virtual qsub_dividable
 {};

/// Tag for compressed matrix in the category lattice
struct compressed2D 
  : virtual matrix, virtual sparse, virtual has_iterator,
    virtual has_cursor
{};

/// Tag for bottom of the category lattice
struct bottom
    : virtual compressed2D, virtual morton_dense, virtual dense2D, 
      virtual dense_col_vector, virtual dense_row_vector
{};

// =====================
// Types for orientation
// =====================


/// Characterizes row-major orientation in matrices and row vector in 1D
struct row_major {};

/// Characterizes column-major orientation in matrices and column vector in 1D
struct col_major {};

} // namespace mtl::tag


/// Characterizes row-major orientation in matrices and row vector in 1D
using tag::row_major;

/// Characterizes column-major orientation in matrices and column vector in 1D
using tag::col_major;

// =====================
// Tags for traversals
// Import some from GLAS
// =====================

namespace tag {

    /// Tag for cursor traversal of non-zero elements of a collection
    /** Also used for elements within rows and columns */
    using glas::tag::nz;

    /// Tag for cursor traversal of all elements of a collection
    /** Also used for elements within rows and columns */
    using glas::tag::all;

    /// Tag for traversal of all rows in a matrix
    using glas::tag::row;
    /// Tag for traversal of all columns in a matrix
    using glas::tag::col;

    /// Tag for traversal of a matrix's major dimension
    /** Is equivalent to glas::tag::row for row-major matrices and 
	glas::tag::col for column-major matrices */
    using glas::tag::major;

    /// Tag for traversal of a matrix's minor dimension
    /** Is equivalent to glas::tag::row for row-major matrices and 
	glas::tag::col for column-major matrices */
    using glas::tag::minor;

    // To define iterators over matrices or rows/cols of it, vectors

    namespace iter {

	/// Tag for iterator traversal of non-zero elements of a collection
	/** Also used for elements within rows and columns */
	struct nz {};

	/// Tag for iterator traversal of all elements of a collection
	/** Also used for elements within rows and columns */
	struct all {};

    } // namespace mtl::tag::iter

    // Same with const iterators

    namespace const_iter {

	/// Tag for const-iterator traversal of non-zero elements of a collection
	/** Also used for elements within rows and columns */
	struct nz {};

	/// Tag for const-iterator traversal of all elements of a collection
	/** Also used for elements within rows and columns */
	struct all {};

    } // namespace mtl::tag::const_iter

} // namespace mtl::tag


} // namespace mtl

#endif // MTL_TAG_INCLUDE
