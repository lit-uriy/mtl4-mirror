// $COPYRIGHT$

#ifndef MTL_DENSE_EL_CURSOR_INCLUDE
#define MTL_DENSE_EL_CURSOR_INCLUDE

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/detail/base_cursor.hpp>

namespace mtl {

/// Cursor over every element of matrix, matrix row/column, or vector
template <typename Value> 
struct dense_el_cursor : public detail::base_cursor<const Value*> 
{
    typedef Value                           value_type;
    typedef const value_type*             const_pointer_type; // ?
    typedef detail::base_cursor<const Value*> super;

    typedef dense_el_cursor               self;

    dense_el_cursor () {} 
    dense_el_cursor (const_pointer_type me) : super(me) {}
    dense_el_cursor (super s) : super(s) {}

    template <typename Parameters>
    dense_el_cursor(dense2D<Value, Parameters> const& ma, size_t r, size_t c)
	: super(ma.elements() + ma.indexer(ma, r, c))
    {}

    self operator+(int x) const
    {
	return self(super::operator+(x));
    }

    int operator-(self const& x)
    {
	return super::operator-(x);
    }
};

} // namespace mtl

#endif // MTL_DENSE_EL_CURSOR_INCLUDE
