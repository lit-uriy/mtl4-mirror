// $COPYRIGHT$

#ifndef MTL_UPDATE_INCLUDE
#define MTL_UPDATE_INCLUDE

namespace mtl { namespace operations {

template <typename Element>
struct update_store
{
    Element& operator() (Element& x, Element const& y)
    {
	return x= y;
    }
};

template <typename Element>
struct update_add
{
    Element& operator() (Element& x, Element const& y)
    {
	return x+= y;
    }
};

template <typename Element>
struct update_mult
{
    Element& operator() (Element& x, Element const& y)
    {
	return x*= y;
    }
};

template <typename Element, typename MonoidOp>
struct update_adapter
{
    Element& operator() (Element& x, Element const& y)
    {
	return x= MonoidOp()(x, y);
    }
};



}} // namespace mtl::operations

#if 0
namespace math {

template <typename Element, typename MonoidOp>
struct identity< Element, mtl::operations::update_adaptor< Element, MonoidOp > >
    : struct identity< Element, MonoidOp >
{};



template < class T >
struct identity< T, mtl::operations::update_store<T> > 
{ 
    static const T value = 0 ; 
    T operator()() const { return value ; }
} ;

template < class T >
const T identity< T, mtl::operations::update_store< T > >::value ;



template < class T >
struct identity< T, mtl::operations::update_add<T> > 
{ 
    static const T value = 0 ; 
    T operator()() const { return value ; }
} ;

template < class T >
const T identity< T, mtl::operations::update_add< T > >::value ;



template < class T >
struct identity< T, mtl::operations::update_mult<T> > 
{ 
    static const T value = 1 ; 
    T operator()() const { return value ; }
} ;

template < class T >
const T identity< T, mtl::operations::update_mult< T > >::value ;

}
#endif

#endif // MTL_UPDATE_INCLUDE
