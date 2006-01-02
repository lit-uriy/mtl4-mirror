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

template <typename Element, typename Updater>
struct update_reference
{
    typedef update_reference   self;

    explicit update_reference(Element& ref) : ref(ref) {}
    
    self& operator= (Element const& val)
    {
	Updater() (ref, val);
	return *this;
    }

    Element&  ref;
};

}} // namespace mtl::operations


namespace math {

// temporary hack, must go to a proper place
template <typename Element, typename MonoidOp>
struct identity {};

#if 0
template <typename Element, typename MonoidOp>
struct identity< Element, mtl::operations::update_adapter< Element, MonoidOp > >
    : struct identity< Element, MonoidOp >
{};
#endif


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

#endif // MTL_UPDATE_INCLUDE
