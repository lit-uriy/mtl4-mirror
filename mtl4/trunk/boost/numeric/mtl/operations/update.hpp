// $COPYRIGHT$

#ifndef MTL_UPDATE_INCLUDE
#define MTL_UPDATE_INCLUDE

namespace mtl { namespace utility {





template <typename Element, typename MonoidOp>
struct update_adaptor
{
    typedef update_adaptor<Element, MonoidOp>   self;

    update_adaptor()                   : op() {}
    update_adaptor(MonoidOp const& op) : op(op) {}
    update_adaptor(self const& s)      : op(s.op) {}

    Element& operator() (Element& x, Element const& y)
    {
	return x= op(x, y);
    }
    MonoidOp    op;
};

}} // namespace mtl::utility

namespace math {

template <typename Element, typename MonoidOp>
struct identity< Element, update_adaptor< Element, MonoidOp > >
    : struct identity< Element, MonoidOp >
{};

}

#endif // MTL_UPDATE_INCLUDE
