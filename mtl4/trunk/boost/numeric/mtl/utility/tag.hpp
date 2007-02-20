// $COPYRIGHT$

#ifndef MTL_TAG_INCLUDE
#define MTL_TAG_INCLUDE

#include <boost/numeric/mtl/utility/glas_tag.hpp>

namespace mtl { namespace tag {

// For non-MTL types not explicitly defined
struct unknown {};

// For internal use (e.g., to invalidate range generators)
struct unsupported {};

// tag for all types
struct universe {};

// tag for any MTL matrix
struct matrix : virtual universe {};

// Tag for any dense MTL matrix
struct dense : virtual universe {};
    
// Tag for matrizes where offet a_ij is x*i + y*j for some x and y
struct has_2D_layout : virtual dense {};
    
// Tag for any sparse MTL matrix
struct sparse : virtual universe {};
    
// Tag for matrices where values are stored contigously in memory
struct contiguous_memory : virtual universe {};

// short cut
struct contiguous_dense : virtual dense, virtual contiguous_memory {};

struct has_iterator : virtual universe {};

// Meaning: is worth unrolling and such
struct has_ra_iterator : virtual has_iterator {};

struct has_fast_ra_iterator : virtual has_ra_iterator {};

struct has_cursor : virtual universe {};

struct has_ra_cursor : virtual has_cursor {};

// Meaning: is worth unrolling and such
struct has_fast_ra_cursor : virtual has_ra_cursor {};

// Only declares that sub_matrix function exist and doesn't say for which ranges it is defined
struct has_sub_matrix : virtual universe {};

// Sub-dividable into quadrants, doesn't require arbitrary ranges to be computable for sub_matrix
struct qsub_dividable : virtual has_sub_matrix {};

// Subdividable, i.e. has sub_matrix function
struct sub_dividable : virtual qsub_dividable {};

// Tags for dispatching on matrix types without dealing 
// with template parameters
struct dense2D 
  : virtual matrix, virtual contiguous_dense, virtual has_fast_ra_cursor, 
    virtual has_fast_ra_iterator, virtual has_2D_layout, virtual sub_dividable
{};

struct morton_dense 
  : virtual matrix, virtual contiguous_dense, virtual has_ra_cursor, 
    virtual has_ra_iterator, virtual qsub_dividable
 {};

struct compressed2D 
  : virtual matrix, virtual sparse, virtual has_iterator,
    virtual has_cursor
{};


// =====================
// Types for orientation
// =====================

// will be exported to ::mtl
struct row_major {};
struct col_major {};


} // namespace mtl::tag


using row_major;
using col_major;

// =====================
// Tags for traversals
// Import some from GLAS
// =====================

namespace tag {

    using glas::tag::nz;
    using glas::tag::all;

    using glas::tag::row;
    using glas::tag::col;

    using glas::tag::major;
    using glas::tag::minor;

    // To define iterators over matrices or rows/cols of it, vectors

    namespace iter {

	struct nz {};
	struct all {};

    } // namespace mtl::tag::iter

    // Same with const iterators

    namespace const_iter {

	struct nz {};
	struct all {};

    } // namespace mtl::tag::const_iter

} // namespace mtl::tag


} // namespace mtl

#endif // MTL_TAG_INCLUDE
