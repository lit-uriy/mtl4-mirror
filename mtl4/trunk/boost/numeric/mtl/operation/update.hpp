// $COPYRIGHT$

#ifndef MTL_UPDATE_INCLUDE
#define MTL_UPDATE_INCLUDE

namespace mtl { namespace operations {

template <typename Element>
struct update_store
{
    template <typename Value>
    Element& operator() (Element& x, Value const& y)
    {
	return x= y;
    }

    // How to fill empty entries; typically directly with /p y
    template <typename Value>
    Element init(Value const& y)
    {
	return y;
    }
};

template <typename Element>
struct update_plus
{
    template <typename Value>
    Element& operator() (Element& x, Value const& y)
    {
	return x+= y;
    }

    // How to fill empty entries; typically directly with /p y
    template <typename Value>
    Element init(Value const& y)
    {
	return y;
    }
};

template <typename Element>
struct update_minus
{
    template <typename Value>
    Element& operator() (Element& x, Value const& y)
    {
	return x-= y;
    }

    // How to fill empty entries. Here the inverse of /p y is needed!!!
    template <typename Value>
    Element init(Value const& y)
    {
	return -y;
    }
};

template <typename Element>
struct update_times
{
    template <typename Value>
    Element& operator() (Element& x, Value const& y)
    {
	return x*= y;
    }

    // How to fill empty entries; typically directly with /p y
    template <typename Value>
    Element init(Value const& y)
    {
	return y;
    }
};

template <typename Element, typename MonoidOp>
struct update_adapter
{
    template <typename Value>
    Element& operator() (Element& x, Value const& y)
    {
	return x= MonoidOp()(x, y);
    }

    // How to fill empty entries
    template <typename Value>
    Element init(Value const& y)
    {
	return y;
    }
};

template <typename Inserter, typename SizeType = std::size_t>
struct update_proxy
{
    typedef update_proxy          self;
    typedef typename Inserter::value_type  value_type;

    explicit update_proxy(Inserter& ins, SizeType row, SizeType col) 
	: ins(ins), row(row), col(col) {}
    
    template <typename Value>
    self& operator<< (Value const& val)
    {
	ins.update (row, col, val);
	return *this;
    }

    Inserter&  ins;
    SizeType   row, col;
};

}} // namespace mtl::operations

#if 0
// inconsistent with linear_algebra/identity.hpp

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
struct identity< T, mtl::operations::update_plus<T> > 
{ 
    static const T value = 0 ; 
    T operator()() const { return value ; }
} ;

template < class T >
const T identity< T, mtl::operations::update_plus< T > >::value ;



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
