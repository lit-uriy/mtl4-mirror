// $COPYRIGHT$

#ifndef MTL_ITERATOR_ADAPTER_INCLUDE
#define MTL_ITERATOR_ADAPTER_INCLUDE

#include <boost/numeric/mtl/utilities/iterator_adapter_detail.hpp>

namespace mtl { namespace utilities {


template <typename PropertyMap, typename Cursor, typename ValueType>
struct const_iterator_adapter
    : public detail::adapter_operators< const_iterator_adapter<PropertyMap, Cursor, ValueType> >
{
    typedef detail::const_iterator_proxy<PropertyMap, Cursor, ValueType>     proxy;

    const_iterator_adapter(PropertyMap const& map, Cursor cursor) 
	: map(map), cursor(cursor) {}

    proxy operator*()
    {
	return proxy(map, cursor);
    }

    PropertyMap const&     map;
    Cursor                 cursor;
};


template <typename PropertyMap, typename Cursor, typename ValueType>
struct iterator_adapter
    : public detail::adapter_operators< iterator_adapter<PropertyMap, Cursor, ValueType> >
{
    typedef detail::iterator_proxy<PropertyMap, Cursor, ValueType>   proxy;

    iterator_adapter(PropertyMap& map, Cursor cursor) 
	: map(map), cursor(cursor) {}

    proxy operator*()
    {
	return proxy(map, cursor);
    }

    PropertyMap&     map;
    Cursor           cursor;
};


}} // namespace mtl::utilities

#endif // MTL_ITERATOR_ADAPTER_INCLUDE
