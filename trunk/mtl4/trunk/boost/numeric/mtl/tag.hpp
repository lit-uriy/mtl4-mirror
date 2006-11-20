// $COPYRIGHT$

#ifndef MTL_TAG_INCLUDE
#define MTL_TAG_INCLUDE

namespace mtl { namespace tag {

// For non-MTL types not explicitly defined
struct unknown {};

// tag for any MTL matrix
struct matrix {};

// Tag for any dense MTL matrix
struct dense : public matrix {};
    
// Tag for any sparse MTL matrix
struct sparse : public matrix {};
    
// Tag for matrices where values are stored contigously in memory
struct contiguous_memory : public matrix {};

struct contiguous_dense : public dense, public contiguous_memory {};

// Tags for dispatching on matrix types without dealing 
// with template parameters
struct dense2D : public contiguous_dense {};

struct morton_dense : public dense {};

struct compressed2D : public sparse {};

// deprecated
// struct fractal : public dense {};


}} // namespace mtl::tag

#endif // MTL_TAG_INCLUDE
