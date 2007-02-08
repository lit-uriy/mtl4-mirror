// $COPYRIGHT$

#ifndef MTL_ITERATOR_ADAPTOR_INCLUDE
#define MTL_ITERATOR_ADAPTOR_INCLUDE

#include <boost/numeric/mtl/utilities/iterator_adaptor_detail.hpp>

namespace mtl { namespace utilities {


// Should be distinguished between random access and forward iterator
// So far all (dense) cursors (within rows/columns) have random access

template <typename PropertyMap, typename Cursor, typename ValueType>
struct const_iterator_adaptor
    : public detail::ra_adaptor_operators< const_iterator_adaptor<PropertyMap, Cursor, ValueType> >
{
    typedef detail::const_iterator_proxy<PropertyMap, Cursor, ValueType>     proxy;

    const_iterator_adaptor(PropertyMap map, Cursor cursor) 
	: map(map), cursor(cursor) {}

    proxy operator*() const
    {
	return proxy(map, cursor);
    }

    PropertyMap            map;
    Cursor                 cursor;
};


template <typename PropertyMap, typename Cursor, typename ValueType>
struct iterator_adaptor
    : public detail::ra_adaptor_operators< iterator_adaptor<PropertyMap, Cursor, ValueType> >
{
    typedef detail::iterator_proxy<PropertyMap, Cursor, ValueType>   proxy;

    iterator_adaptor(PropertyMap map, Cursor cursor) 
	: map(map), cursor(cursor) {}

    proxy operator*()
    {
	return proxy(map, cursor);
    }

    PropertyMap      map;
    Cursor           cursor;
};


}} // namespace mtl::utilities

#endif // MTL_ITERATOR_ADAPTOR_INCLUDE
