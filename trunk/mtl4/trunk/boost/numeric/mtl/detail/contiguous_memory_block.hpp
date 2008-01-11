// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

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

// Macro MTL_ENABLE_ALIGNMENT is by default not set

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
#     ifndef MTL_ENABLE_ALIGNMENT
	this->data= new value_type[size];
#     else
	bool        align= size * sizeof(value_type) >= MTL_ALIGNMENT_LIMIT;
	std::size_t bytes= size * sizeof(value_type);

	if (align)
	    bytes+= MTL_ALIGNMENT - 1;

	char* p= this->malloc_address= new char[bytes];
	if (align)
	    while ((long int)(p) % MTL_ALIGNMENT) p++;

	this->data= reinterpret_cast<value_type*>(p);
#     endif
    }

    void delete_it()
    {
	// printf("delete_it: data %p, malloc %p\n", this->data, malloc_address);      
#       ifndef MTL_ENABLE_ALIGNMENT
	    if (!extern_memory && this->data) delete[] this->data;
#       else
	    if (!extern_memory && malloc_address) delete[] malloc_address;
#       endif
    }

  public:
    generic_array()
	: extern_memory(false), 
#       ifdef MTL_ENABLE_ALIGNMENT
	  malloc_address(0), 
#       endif
	  data(0) 
    {}

    explicit generic_array(Value *data) 
	: extern_memory(true), 
#       ifdef MTL_ENABLE_ALIGNMENT
	  malloc_address(0), 
#       endif
	  data(data) 
    {}    

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
	delete_it();
	alloc(size);
    }

    ~generic_array()
    {
	delete_it();
    }

    void swap(self& other)
    {
	using std::swap;
	swap(extern_memory, other.extern_memory);
#       ifdef MTL_ENABLE_ALIGNMENT
	    swap(malloc_address, other.malloc_address);
#       endif
	swap(data, other.data);
    }	

  protected:
    bool                                      extern_memory;       // whether pointer to external data or own
#ifdef MTL_ENABLE_ALIGNMENT
    char*                                     malloc_address;
#endif
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

    contiguous_memory_block() : my_used_memory(0) {}

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
