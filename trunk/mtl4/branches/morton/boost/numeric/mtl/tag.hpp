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
    
// Tags for dispatching on matrix types without dealing 
// with template parameters
struct dense2D : public dense {};
struct fractal : public dense {};

struct morton_dense : public dense {};

}} // namespace mtl

#endif // MTL_TAG_INCLUDE
