// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_UPDATE_INCLUDE
#define MTL_UPDATE_INCLUDE

#include <boost/numeric/mtl/operation/assign_mode.hpp>
#include <boost/numeric/mtl/utility/range_generator.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/ashape.hpp>

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
	return lshift(val, typename ashape::ashape<Value>::type());
    }

    template <typename Value>
    self& operator= (Value const& val)
    {
	ins.template modify<update_store<value_type> > (row, col, val);
	return *this;
    }

    template <typename Value>
    self& operator+= (Value const& val)
    {
	ins.template modify<update_plus<value_type> > (row, col, val);
	return *this;
    }

  private:

    // Update scalar value as before
    template <typename Value>
    self& lshift (Value const& val, ashape::scal)
    {
	ins.update (row, col, val);
	return *this;
    }
    
    typedef typename ashape::ashape<typename Inserter::matrix_type>::type shape_type;

    // Update an entire matrix considered as block
    template <typename MatrixSrc>
    self& lshift (const MatrixSrc& src, shape_type)
    {
	typename traits::row<MatrixSrc>::type             row(src); 
	typename traits::col<MatrixSrc>::type             col(src); 
	typename traits::const_value<MatrixSrc>::type     value(src); 

	typedef typename traits::range_generator<tag::major, MatrixSrc>::type  cursor_type;
	typedef typename traits::range_generator<tag::nz, cursor_type>::type   icursor_type;
	
	for (cursor_type cursor = begin<tag::major>(src), cend = end<tag::major>(src); cursor != cend; ++cursor) 	    
	    for (icursor_type icursor = begin<tag::nz>(cursor), icend = end<tag::nz>(cursor); icursor != icend; ++icursor)
		ins.update(row(*icursor) + this->row, col(*icursor) + this->col, value(*icursor));

	return *this;
    }

    Inserter&  ins;
    SizeType   row, col;
};

/// Compute updater that corresponds to assign_mode
template <typename Assign, typename Value>
struct update_assign_mode {};

template <typename Value>
struct update_assign_mode<assign::assign_sum, Value>
{
    typedef update_plus<Value> type;
};

template <typename Value>
struct update_assign_mode<assign::plus_sum, Value>
{
    typedef update_plus<Value> type;
};

template <typename Value>
struct update_assign_mode<assign::minus_sum, Value>
{
    typedef update_minus<Value> type;
};

} // namespace operations

using operations::update_store;
using operations::update_plus;
using operations::update_minus;
using operations::update_times;

} // namespace mtl




















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
