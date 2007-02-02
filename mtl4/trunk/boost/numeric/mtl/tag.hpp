// $COPYRIGHT$

#ifndef MTL_TAG_INCLUDE
#define MTL_TAG_INCLUDE

namespace mtl { namespace tag {

// For non-MTL types not explicitly defined
struct unknown {};

// tag for all types
struct universe {};

// tag for any MTL matrix
struct matrix : virtual universe {};

// Tag for any dense MTL matrix
struct dense : virtual universe {};
    
// Tag for any sparse MTL matrix
struct sparse : virtual universe {};
    
// Tag for matrices where values are stored contigously in memory
struct contiguous_memory : virtual universe {};

// short cut
struct contiguous_dense : virtual dense, virtual contiguous_memory {};

struct has_iterator : virtual universe {};

struct has_ra_iterator : virtual has_iterator {};

struct has_fast_ra_iterator : virtual has_ra_iterator {};

struct has_cursor : virtual universe {};

struct has_ra_cursor : virtual has_cursor {};

struct has_fast_ra_cursor : virtual has_ra_cursor {};

// Tags for dispatching on matrix types without dealing 
// with template parameters
struct dense2D 
  : virtual matrix, virtual contiguous_dense, virtual has_fast_ra_cursor, 
    virtual has_fast_ra_iterator
{};

struct morton_dense 
  : virtual matrix, virtual contiguous_dense, virtual has_ra_cursor, 
    virtual has_ra_iterator
 {};

struct compressed2D 
  : virtual matrix, virtual sparse, virtual has_iterator,
    virtual has_cursor
{};

// deprecated
// struct fractal : virtual dense {};


}} // namespace mtl::tag

#endif // MTL_TAG_INCLUDE
