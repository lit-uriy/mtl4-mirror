// $COPYRIGHT$

#ifndef MTL_STRIDED_BASE_CURSOR_INCLUDE
#define MTL_STRIDED_BASE_CURSOR_INCLUDE

#include <boost/numeric/mtl/detail/base_cursor.hpp>

namespace mtl { namespace detail {

template <class Key> struct strided_base_cursor 
 : base_cursor<Key>
{
    typedef Key                  key_type;
    typedef base_cursor<Key>     base;
    typedef strided_base_cursor  self;

    strided_base_cursor () {} 
    strided_base_cursor (key_type key, std::size_t stride) 
	: base(key), stride(stride) 
    {}

    self& operator++ () 
    { 
	this->key+= stride; return *this; 
    }
    self operator++ (int) 
    { 
	self tmp = *this; 
	this->key+= stride; 
	return tmp; 
    }
    self& operator-- () 
    { 
	this->key-= stride; 
	return *this; 
    }
    self operator-- (int) 
    { 
	self tmp = *this; 
	this->key-= stride; 
	return tmp; 
    }
    self& operator+=(int n) 
    { 
	this->key += stride * n; 
	return *this; 
    }
    self operator+(int n) const
    {
	self tmp(*this);
	tmp+= n;
	return tmp;
    }
    self& operator-=(int n) 
    { 
	this->key -= stride * n; 
	return *this; 
    }
    std::size_t stride;
};

}} // namespace mtl::detail

#endif // MTL_STRIDED_BASE_CURSOR_INCLUDE
