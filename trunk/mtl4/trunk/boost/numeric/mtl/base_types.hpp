// $COPYRIGHT$

#ifndef MTL_BASE_TYPES_INCLUDE
#define MTL_BASE_TYPES_INCLUDE

#include <boost/type_traits.hpp>

namespace mtl {

// typetrait to check if certain type is MTL type
// then it provides additional types and variables
template <class T> struct is_mtl_type 
{ 
    static bool const value= false; 
};

// Types for orientation
struct row_major {};
struct col_major {};

// Orientation type for transposed matrix
template <class T> struct transposed_orientation {};

template<> struct transposed_orientation<row_major> 
{
    typedef col_major type; 
};

template<> struct transposed_orientation<col_major> 
{
    typedef row_major type; 
};

namespace tag 
{

    // tag for any MTL matrix
    struct matrix_tag {};

    // Tag for any dense MTL matrix
    struct dense_tag : public matrix_tag {};
    
    // Tag for any sparse MTL matrix
    struct sparse_tag : public matrix_tag {};
    
    // Tags for dispatching on matrix types without dealing 
    // with template parameters
    struct dense2D_tag : public dense_tag {};
    struct fractal_tag : public dense_tag {};
} //  namespace tag 


// Get matrix tag for dispatching
// Has to be specialized for each matrix
template <class Matrix> struct matrix_category {};

} // namespace mtl

#endif // MTL_BASE_TYPES_INCLUDE




