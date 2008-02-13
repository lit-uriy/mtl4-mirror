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
#include <adobe/move.hpp>

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


// Manage size only if template parameter is 0
template <unsigned Size>
struct size_helper
{
    typedef size_helper self;

    explicit size_helper(std::size_t size)
    {
	set_size(size);
    }

    void set_size(std::size_t size)
    {
#     ifndef MTL_IGNORE_STATIC_SIZE_VIOLATION
	MTL_THROW_IF(Size != size, change_static_size());
#     endif
    }

    std::size_t used_memory() const
    {
	return Size;
    }

    void swap(self& other) const {}
};

template <>
struct size_helper<0>
{
    typedef size_helper self;

    size_helper(std::size_t size= 0) : my_used_memory(size) {}

    void set_size(std::size_t size)
    {
	my_used_memory= size;
    }

    std::size_t used_memory() const
    {
	return my_used_memory;
    }

    void swap(self& other) 
    {
	std::swap(my_used_memory, other.my_used_memory);
    }

  protected:
    std::size_t                               my_used_memory;
};


// Encapsulate behavior of alignment

# ifdef MTL_ENABLE_ALIGNMENT

    template <typename Value>
    struct alignment_helper
    {
	typedef alignment_helper self;

	alignment_helper() : malloc_address(0) {}

	Value* alligned_alloc(std::size_t size)
	{
	    bool        align= size * sizeof(value_type) >= MTL_ALIGNMENT_LIMIT;
	    std::size_t bytes= size * sizeof(value_type);
	    
	    if (align)
		bytes+= MTL_ALIGNMENT - 1;

	    char* p= malloc_address= new char[bytes];
	    if (align)
		while ((long int)(p) % MTL_ALIGNMENT) p++;

	    return reinterpret_cast<value_type*>(p);
	}

	void aligned_delete(bool extern_memory, Value*& data)
	{
	    if (!extern_memory && malloc_address) delete[] malloc_address;
	    data= 0;
	}

	void swap(self& other) 
	{
	    swap(malloc_address, other.malloc_address);
	}

      private:	    
	char*                                     malloc_address;
    };

# else

    template <typename Value>
    struct alignment_helper
    {
	typedef alignment_helper self;

	Value* alligned_alloc(std::size_t size)
	{
	    return new Value[size];
	}

	void aligned_delete(bool extern_memory, Value*& data)
	{
	    if (!extern_memory && data) delete[] data;
	}

	void swap(self& other) const {}
    };

# endif


template <typename Value, bool OnStack, unsigned Size>
struct memory_crtp
//    : public contiguous_memory_block<Value, OnStack, Size>
{
    typedef contiguous_memory_block<Value, OnStack, Size> base;

    static bool const                         on_stack= OnStack;
    
    typedef Value                             value_type;
    typedef value_type*                       pointer_type;
    typedef const value_type*                 const_pointer_type;

    // offset of key (pointer) w.r.t. data 
    // values must be stored consecutively
    size_t offset(const Value* p) const 
    { 
      return p - static_cast<const base&>(*this).data; 
    }

    // returns pointer to data
    pointer_type elements()
    {
      return static_cast<base&>(*this).data; 
    }

    // returns const pointer to data
    const_pointer_type elements() const 
    {
      return static_cast<const base&>(*this).data; 
    }

    // returns n-th value in consecutive memory
    // (whatever this means in the corr. matrix format)
    value_type& value_n(size_t offset)
    { 
      return static_cast<base&>(*this).data[offset]; 
    }

    // returns n-th value in consecutive memory
    // (whatever this means in the corr. matrix format)
    const value_type& value_n(size_t offset) const 
    { 
      return static_cast<const base&>(*this).data[offset]; 
    }
    
};


template <typename Value, bool OnStack, unsigned Size>
struct contiguous_memory_block
    : public size_helper<Size>,
      public alignment_helper<Value>,
      public memory_crtp<Value, OnStack, Size>
{
    typedef Value                             value_type;
    typedef contiguous_memory_block                     self;
    typedef size_helper<Size>                 size_base;
    typedef alignment_helper<Value>           alignment_base;

  private:
    void alloc(std::size_t size)
    {
	this->set_size(size);
	data= this->alligned_alloc(this->used_memory());
    }

    void delete_it()
    {
	this->aligned_delete(extern_memory, data);
    }

  public:
    contiguous_memory_block() : extern_memory(false), data(0) {}

    explicit contiguous_memory_block(Value *data, std::size_t size) 
	: extern_memory(true), size_base(size), data(data)
    {}    

    explicit contiguous_memory_block(std::size_t size) : extern_memory(false)
    {
	std::cout << "Constructor with size.\n";
	alloc(size);
    }


    // If possible move data
    explicit contiguous_memory_block(self& other, adobe::move_ctor)
	: extern_memory(false), data(0)
    {
	std::cout << "Ich habe gemovet.\n";
	swap(*this, other);
    }

    // Default copy constructor
    contiguous_memory_block(const self& other)
    {
	std::cout << "Ich habe kopiert (von gleichem array-Typ).\n";
	
	alloc(other.used_memory());
	std::copy(other.data, other.data + other.used_memory(), data);
    }

    // Other types must be copied always
    template<typename Value2, bool OnStack2, unsigned Size2>
    explicit contiguous_memory_block(const contiguous_memory_block<Value2, OnStack2, Size2>& other)
	: extern_memory(false)
    {
	std::cout << "Ich habe kopiert (von anderem array-Typ).\n";
	
	alloc(other.used_memory());
	std::copy(other.data, other.data + other.used_memory(), data);
    }


    // Operator takes parameter by value and consumes it
    self& operator=(self other)
    {
	std::cout << "Konsumierender Zuweisungsoperator.\n";
	swap(other);
	return *this;
    }

    template<typename Value2, bool OnStack2, unsigned Size2>
    self& operator=(const contiguous_memory_block<Value2, OnStack2, Size2>& other)
    {
	std::cout << "Zuweisung von anderem array-Typ -> Kopieren.\n";
	MTL_DEBUG_THROW_IF(this->used_memory() != other.used_memory(), incompatible_size());
	std::copy(other.data, other.data + other.used_memory(), data);
    }


    void realloc(std::size_t size)
    {
	// If already have memory of the right size we can keep it
	if (size == this->used_memory()) 
	    return;
	MTL_THROW_IF(extern_memory, 
		     logic_error("Can't change the size of collections with external memory"));
	delete_it();
	alloc(size);
    }

    ~contiguous_memory_block()
    {
	//std::cout << "Delete block with address " << data << '\n';
	delete_it();
    }

    void swap(self& other)
    {
	using std::swap;
	swap(extern_memory, other.extern_memory);
	swap(data, other.data);
	size_base::swap(other);
	alignment_base::swap(other);
	
    }	

  protected:
    bool                                      extern_memory;       // whether pointer to external data or own
  public:
    Value                                     *data;
};

template <typename Value, unsigned Size>
struct contiguous_memory_block<Value, true, Size>
{
    typedef Value                             value_type;
    typedef contiguous_memory_block                     self;

    Value    data[Size];
    explicit contiguous_memory_block(std::size_t) {}

    // Move-semantics ignored for arrays on stack
    contiguous_memory_block(const self& other)
    {
	std::cout << "Ich habe kopiert (von gleichem array-Typ).\n";
	std::copy(other.data, other.data+Size, data);
    }


    template<typename Value2, bool OnStack2, unsigned Size2>
    explicit contiguous_memory_block(const contiguous_memory_block<Value2, OnStack2, Size2>& other)
    {
	std::cout << "Ich habe kopiert (von anderem array-Typ).\n";
	MTL_DEBUG_THROW_IF(Size != other.used_memory(), incompatible_size());
	std::copy(other.data, other.data + other.used_memory(), data);
    }

    self& operator=(const self& other)
    {
	std::cout << "Zuweisung (von gleichem array-Typ).\n";
	std::copy(other.data, other.data+Size, data);
	return *this;
    }

    template<typename Value2, bool OnStack2, unsigned Size2>
    self& operator=(const contiguous_memory_block<Value2, OnStack2, Size2>& other)
    {
	std::cout << "Ich habe kopiert (von anderem array-Typ).\n";
	MTL_DEBUG_THROW_IF(Size != other.used_memory(), incompatible_size());
	std::copy(other.data, other.data + other.used_memory(), data);
    }


    void swap (self& other)
    {
	using std::swap;
	swap(*this, other);
    }


    void realloc(std::size_t) 
    {
	assert(false); // Arrays on stack cannot be reallocated
    }

    std::size_t used_memory() const
    {
	return Size;
    }
};




#if 0

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

  public:
    // Reference to external data (must be heap)
    explicit contiguous_memory_block(value_type* a, std::size_t size) : base(a, size) {}

    explicit contiguous_memory_block(std::size_t size) : base(size) {}

    explicit contiguous_memory_block(self& other, adobe::move_ctor)
	: base(other, adobe::move_ctor())
    {}

    contiguous_memory_block(const self& other) : base(other) {}

    template<typename Value2, bool OnStack2, unsigned Size2>
    explicit contiguous_memory_block(const contiguous_memory_block<Value2, OnStack2, Size2>& other)
	: base(other) {}

    // Inherited assignment operators return wrong type
    // But one is called by value, the other isn't
    // self& operator=(const self

    template<typename Value2, bool OnStack2, unsigned Size2>
    self& operator=(const contiguous_memory_block<Value2, OnStack2, Size2>& other)
    {
	base::operator=(other);
	return *this;
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

};

#endif

}} // namespace mtl::detail

namespace adobe {
#if 0
    template <typename Value, bool OnStack, unsigned Size>
    struct is_movable< mtl::detail::generic_array<Value, OnStack, Size> > : boost::mpl::bool_<!OnStack> {};
#endif
    template <typename Value, bool OnStack, unsigned Size>
    struct is_movable< mtl::detail::contiguous_memory_block<Value, OnStack, Size> > : boost::mpl::bool_<!OnStack> {};
}

#endif // MTL_CONTIGUOUS_MEMORY_BLOCK_INCLUDE
