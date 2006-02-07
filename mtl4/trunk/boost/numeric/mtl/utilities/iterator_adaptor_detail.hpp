// $COPYRIGHT$

#ifndef MTL_ITERATOR_ADAPTOR_DETAIL_INCLUDE
#define MTL_ITERATOR_ADAPTOR_DETAIL_INCLUDE

namespace mtl { namespace utilities { namespace detail {


template <typename Adaptor>
struct adaptor_operators
{
    Adaptor& operator++() 
    {
	Adaptor& me = static_cast<Adaptor&>(*this);
	++me.cursor;
	return me;
    }

    Adaptor& operator++(int) 
    {
	Adaptor& me = static_cast<Adaptor&>(*this);
	Adaptor  tmp(me);
	++me.cursor;
	return tmp;
    }
    
    bool operator==(Adaptor const& x) const
    {
	Adaptor const& me = static_cast<Adaptor const&>(*this);

	// Compare addresses of property maps
	return &me.map == &x.map && me.cursor == x.cursor;

	// Certainly better they provide comparison
	// return me.map == x.map && me.cursor == x.cursor; 
    }

    bool operator!=(Adaptor const& x) const
    {
	return !operator==(x);
    }
};


template <typename PropertyMap, typename Cursor, typename ValueType>
struct const_iterator_proxy
{
    const_iterator_proxy(PropertyMap const& map, Cursor cursor) 
	: map(map), cursor(cursor) {}

    operator ValueType() const
    {
	return map(*cursor);
    }

    PropertyMap const&     map;
    Cursor                 cursor;
};


template <typename PropertyMap, typename Cursor, typename ValueType>
struct iterator_proxy
{
    typedef iterator_proxy                    self;

    iterator_proxy(PropertyMap& map, Cursor cursor) 
	: map(map), cursor(cursor) {}

    operator ValueType() const
    {
	return map(*cursor);
    }

    self& operator=(ValueType const& value)
    {
	map(*cursor, value);
	return *this;
    }

    PropertyMap&           map;
    Cursor                 cursor;
};

}}} // namespace mtl::utilities::detail

#endif // MTL_ITERATOR_ADAPTOR_DETAIL_INCLUDE
