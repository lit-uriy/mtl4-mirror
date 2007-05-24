// $COPYRIGHT$

#ifndef MTL_CONTIGUOUS_MEMORY_BLOCK_INCLUDE
#define MTL_CONTIGUOUS_MEMORY_BLOCK_INCLUDE

#include <cassert>
#include <algorithm>
#include <boost/static_assert.hpp>
#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/matrix/dimension.hpp>
#include <boost/numeric/mtl/detail/index.hpp>

namespace mtl { namespace detail {
using std::size_t;
  
template <typename Matrix, bool Enable>
struct array_size
{
    // More convenient when always exist (and then not use it)
    static std::size_t const value= 0;
};

// Minimal size of memory allocation using alignment
#ifndef MTL_ALIGNMENT_LIMIT
#  define MTL_ALIGNMENT_LIMIT 1024
#endif

// Alignment in memory
#ifndef MTL_ALIGNMENT
#  define MTL_ALIGNMENT 128
#endif

template <typename Value, bool OnStack, unsigned Size= 0>
struct generic_array
{
    typedef Value                             value_type;
    typedef generic_array                     self;

    void alloc(std::size_t size)
    {
	if (size * sizeof(value_type) >= MTL_ALIGNMENT_LIMIT) {
	    char* p= this->malloc_address= new char[size * sizeof(value_type) + MTL_ALIGNMENT - 1];
	    while ((long int)(p) % MTL_ALIGNMENT) p++;
	    this->data= reinterpret_cast<value_type*>(p);
	} else {
	    // malloc_address= new char[size * sizeof(value_type)];
	    // this->data= reinterpret_cast<value_type*>(malloc_address);
	    this->data= new value_type[size];
	    malloc_address= reinterpret_cast<char*>(this->data);
	}

    }

  public:
    generic_array(): extern_memory(true), malloc_address(0), data(0) {}

    explicit generic_array(Value *data) : extern_memory(true), malloc_address(0), data(data) {}    

    explicit generic_array(std::size_t size) : extern_memory(false)
    {
	alloc(size);
    }

    void realloc(std::size_t size, std::size_t old_size)
    {
	// If already have memory of the right size we can keep it
	if (size == old_size) 
	    return;
	MTL_THROW_IF(extern_memory, 
		     logic_error("Can't change the size of collections with external memory"));
	// Free old memory (if allocated)
	if (!extern_memory && malloc_address) {
	    // printf("realloc: data %p, malloc %p\n", this->data, malloc_address);      
	    delete[] malloc_address; }
	alloc(size);
    }

    ~generic_array()
    {
	// printf("destructor: data %p, malloc %p\n", this->data, malloc_address);      
	if (!extern_memory && malloc_address) delete[] malloc_address;
    }

    void swap(self& other)
    {
	using std::swap;
	swap(extern_memory, other.extern_memory);
	swap(malloc_address, other.malloc_address);
	swap(data, other.data);
    }	

  protected:
    bool                                      extern_memory;       // whether pointer to external data or own
    char*                                     malloc_address;
  public:
    Value    *data;
};

template <typename Value, unsigned Size>
struct generic_array<Value, true, Size>
{
    Value    data[Size];
    explicit generic_array(std::size_t) {}

    void swap (generic_array<Value, true, Size>& other)
    {
	using std::swap;
	swap(*this, other);
    }

    void realloc(std::size_t) 
    {
	assert(false); // Arrays on stack cannot be reallocated
    }
};

// Base class for matrices that have contigous piece of memory
template <typename Value, bool OnStack, unsigned Size>
struct contiguous_memory_block 
  : public generic_array<Value, OnStack, Size>
{
    typedef contiguous_memory_block             self;
    typedef generic_array<Value, OnStack, Size> base;

    static bool const                         on_stack= OnStack;
    
    typedef Value                             value_type;
    typedef value_type*                       pointer_type;
    typedef const value_type*                 const_pointer_type;

  protected:
    std::size_t                               my_used_memory;

  public:
    // Reference to external data (must be heap)
    explicit contiguous_memory_block(value_type* a, std::size_t size)
      : base(a), my_used_memory(size) {}

    explicit contiguous_memory_block(std::size_t size)
	: base(size), my_used_memory(size) {}

    void swap(self& other)
    {
	base::swap(other);
	std::swap(my_used_memory, other.my_used_memory);
    }

    void realloc(std::size_t size) 
    {
	base::realloc(size, my_used_memory);
	my_used_memory= size;
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
    value_type& value_n(size_t offset)
    { 
      return this->data[offset]; 
    }

    // returns n-th value in consecutive memory
    // (whatever this means in the corr. matrix format)
    const value_type& value_n(size_t offset) const 
    { 
      return this->data[offset]; 
    }

    std::size_t used_memory() const
    {
	return my_used_memory;
    }
};

}} // namespace mtl::detail

#endif // MTL_CONTIGUOUS_MEMORY_BLOCK_INCLUDE
