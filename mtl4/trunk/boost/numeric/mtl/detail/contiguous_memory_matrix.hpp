// $COPYRIGHT$

#ifndef MTL_CONTIGUOUS_MEMORY_MATRIX_INCLUDE
#define MTL_CONTIGUOUS_MEMORY_MATRIX_INCLUDE

#include <boost/static_assert.hpp>
#include <boost/numeric/mtl/base_types.hpp>
#include <boost/numeric/mtl/dimensions.hpp>
#include <boost/numeric/mtl/index.hpp>

namespace mtl { namespace detail {
using std::size_t;
  
// Base class for matrices that have contigous piece of memory
template <class Elt, class Parameters>
struct contiguous_memory_matrix 
{
    typedef typename Parameters::orientation  orientation;
    typedef typename Parameters::index        index_type;
    typedef typename Parameters::dimensions   dim_type;
    typedef Elt                     value_type;
    typedef value_type*             pointer_type;
    typedef const value_type*       const_pointer_type;
    typedef pointer_type            key_type;
  protected:
    bool                            ext;       // whether pointer to external data or own

  public:
    explicit contiguous_memory_matrix(value_type* a)
      : data(a), ext(true) {}

    explicit contiguous_memory_matrix(std::size_t size)
	: ext(false)
    {
	data = new value_type[size];
    }

    ~contiguous_memory_matrix()
    {
	if (!ext && data) delete[] data;
    }

    // offset of key (pointer) w.r.t. data 
    // values must be stored consecutively
    size_t offset(const value_type* p) const 
    { 
      return p - data; 
    }

    // returns pointer to data
    pointer_type elements()
    {
      return data; 
    }

    // returns const pointer to data
    const_pointer_type elements() const 
    {
      return data; 
    }

    // returns n-th value in consecutive memory
    // (whatever this means in the corr. matrix format)
    value_type value_n(size_t offset) const 
    { 
      return data[offset]; 
    }

  protected:
    value_type*                     data;      // pointer to matrix
};

}} // namespace mtl::detail

#endif // MTL_CONTIGUOUS_MEMORY_MATRIX_INCLUDE
