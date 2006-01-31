// $COPYRIGHT$

#ifndef MTL_CONTIGUOUS_MEMORY_MATRIX_INCLUDE
#define MTL_CONTIGUOUS_MEMORY_MATRIX_INCLUDE

#include <boost/static_assert.hpp>
#include <boost/numeric/mtl/base_types.hpp>
#include <boost/numeric/mtl/dimensions.hpp>
#include <boost/numeric/mtl/index.hpp>

namespace mtl { namespace detail {
using std::size_t;
  
template <typename Matrix, bool Enable>
struct array_size
{
    // More convenient when always exist (and then not use it)
    static std::size_t const value= 0;
};

template <typename Elt, bool OnStack, unsigned Size= 0>
struct generic_array
{
    Elt    *data;
};

template <typename Elt, unsigned Size>
struct generic_array<Elt, true, Size>
{
    Elt    data[Size];
};


// Base class for matrices that have contigous piece of memory
template <typename Elt, bool OnStack, unsigned Size= 0>
struct contiguous_memory_matrix 
  : public generic_array<Elt, OnStack, Size>
{
    typedef generic_array<Elt, OnStack, Size>       base;

    static bool const                         on_stack= OnStack;
    
    typedef Elt                               value_type;
    typedef value_type*                       pointer_type;
    typedef const value_type*                 const_pointer_type;

  protected:
    bool                            ext;       // whether pointer to external data or own

  public:
    // Reference to external data (must be heap)
    explicit contiguous_memory_matrix(value_type* a)
      : base::data(a), ext(true) {}

    explicit contiguous_memory_matrix(std::size_t size)
	: ext(false)
    {
	if (!on_stack) this->data = new value_type[size];
    }

    ~contiguous_memory_matrix()
    {
	if (!on_stack && !ext && data) delete[] this->data;
    }

    // offset of key (pointer) w.r.t. data 
    // values must be stored consecutively
    size_t offset(const value_type* p) const 
    { 
      return p - this->data; 
    }

    // returns pointer to data
    pointer_type elements()
    {
      return this->data; 
    }

    // returns const pointer to data
    const_pointer_type elements() const 
    {
      return this->data; 
    }

    // returns n-th value in consecutive memory
    // (whatever this means in the corr. matrix format)
    value_type value_n(size_t offset) const 
    { 
      return this->data[offset]; 
    }

};

}} // namespace mtl::detail

#endif // MTL_CONTIGUOUS_MEMORY_MATRIX_INCLUDE
