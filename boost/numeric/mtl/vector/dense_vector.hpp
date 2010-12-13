// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University.
//               2008 Dresden University of Technology and the Trustees of Indiana University.
//               2010 SimuNova UG (haftungsbeschränkt), www.simunova.com.
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

// Adapted from GLAS implementation by Karl Meerbergen and Toon Knappen


#ifndef MTL_DENSE_VECTOR_INCLUDE
#define MTL_DENSE_VECTOR_INCLUDE


#include <iostream>
#include <cassert>
#include <vector>
#include <algorithm>
#include <boost/static_assert.hpp>
#include <boost/utility/enable_if.hpp>

#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/utility/ashape.hpp>
#include <boost/numeric/mtl/vector/all_vec_expr.hpp>
#include <boost/numeric/mtl/vector/parameter.hpp>
#include <boost/numeric/mtl/detail/contiguous_memory_block.hpp>
#include <boost/numeric/mtl/vector/crtp_base_vector.hpp>
#include <boost/numeric/mtl/utility/dense_el_cursor.hpp>
#include <boost/numeric/mtl/utility/range_generator.hpp>
#include <boost/numeric/mtl/utility/irange.hpp>
#include <boost/numeric/mtl/utility/is_static.hpp>
#include <boost/numeric/mtl/utility/is_row_major.hpp>
#include <boost/numeric/mtl/utility/transposed_orientation.hpp>
#include <boost/numeric/mtl/operation/is_negative.hpp>


namespace mtl { namespace vector {

/// Dense vector
template <class Value, typename Parameters = parameters<> >
class dense_vector
  : public vec_expr<dense_vector<Value, Parameters> >,
    public ::mtl::detail::contiguous_memory_block< Value, Parameters::on_stack, Parameters::dimension::value >,
    public crtp_base_vector< dense_vector<Value, Parameters>, Value, std::size_t >
{
    typedef dense_vector                                                             self;
    typedef ::mtl::detail::contiguous_memory_block< Value, Parameters::on_stack, 
                                                    Parameters::dimension::value >   memory_base;
    typedef crtp_base_vector< self, Value, std::size_t >                             crtp_base;
    typedef crtp_vector_assign< self, Value, std::size_t >                           assign_base;
    typedef vec_expr<dense_vector<Value, Parameters> >                               expr_base;
  public:
    typedef Value             value_type ; 
    typedef std::size_t       size_type ;
    typedef value_type&       reference ;
    typedef value_type const& const_reference ;
    typedef Value*            pointer ;
    typedef Value const*      const_pointer ;
    typedef typename Parameters::orientation  orientation;

    typedef const_pointer     key_type;
    
    /// Check whether index is non-negative and less than size
    void check_index( size_type i ) const
    {
	MTL_DEBUG_THROW_IF( is_negative(i) ||  i >= size(*this), index_out_of_range());
    }

    /// Check for a given vector if the sizes are equal or this has size 0 (and can take the size of source)
    void check_dim( size_type s ) const
    {
	MTL_DEBUG_THROW_IF( size(*this) != 0 && size(*this) != s, incompatible_size());
    }

    /// Check at compile time for a given vector if the sizes are equal
    void static_check( size_type s) const
    {
	assert(!traits::is_static<self>::value || s == size(typename Parameters::dimension()));
    }

    /// Check if a given vector expression whether it has the same shape as the object
    /** Both must be row vectors or column vectors. The elements must have the same
	algebraic shape, e.g. a row vector of scalars is not compatible with a row
	vector of matrices. **/
    template <class E>
    void check_consistent_shape( vec_expr<E> const& e ) const
    {
	MTL_DEBUG_THROW_IF((!boost::is_same<
			        typename ashape::ashape<self>::type
			      , typename ashape::ashape<E>::type
			    >::value),
			   incompatible_shape());
    }

    /// Default constructor
    dense_vector( ) : memory_base( Parameters::dimension::value ) {}
    
    /// Constructor for size \p n
    explicit dense_vector( size_type n ) : memory_base( n ) { static_check( n ); }
    
    /// Constructor for size \p n and value \p value that is set in all entries
    explicit dense_vector( size_type n, value_type value )
      : memory_base( n ) 
    {
	static_check( n );
	std::fill(begin(), end(), value);
    }

    /// Constructor for size \p n and \p address
    /** Can be used to handle vectors from other libraries, even in other languages like Fortran **/
    explicit dense_vector( size_type n, value_type *address )
      : memory_base( address, n, true ) 
    { static_check( n ); }

    /// Copy constructor
    dense_vector( const self& src )
      : memory_base( src ) 
    {
	using std::copy;
	copy(src.begin(), src.end(), begin());
    }

    dense_vector( const self& src, clone_ctor )
      : memory_base( src, clone_ctor()) {} 

    /// Constructor from vector expressions
    template <typename VectorSrc>
    explicit dense_vector(const VectorSrc& src,
			  typename boost::disable_if<boost::is_integral<VectorSrc>, int >::type= 0)
    {	*this= src;    }

    /// Size of v (like a free function)
    friend inline size_type size(const self& v)  { return v.used_memory() ; }
    
    /// Stride is always 1 
    size_type stride() const { return 1 ; }

    /// i-th entry
    reference operator()( size_type i ) 
    {
        check_index(i);
        return this->value_n( i ) ;
    }

    /// i-th entry (constant)
    const_reference operator()( size_type i ) const 
    {
        check_index(i);
        return this->value_n( i ) ;
    }

    reference operator[]( size_type i ) { return (*this)( i ) ; } ///< i-th entry
    const_reference operator[]( size_type i ) const { return (*this)( i ) ;  } ///< i-th entry (constant)

    self operator[]( irange r ) { return sub_vector(*this, r.start(), r.finish()); } ///< sub-vector
    const self  operator[]( irange r ) const { return sub_vector(*this, r.start(), r.finish());  } ///< sub-vector
    
    void delay_assign() const {}

    // Compatibility with STL
    const_pointer begin() const { return this->elements() ; }
    const_pointer end() const { return this->elements() + size(*this) ; }    
    pointer begin() { return this->elements() ; }
    pointer end() { return this->elements() + size(*this) ; }
    bool empty() const { return size(*this) == 0; }


    /// Address of first data entry; to be used with care.
    value_type* address_data() { return begin(); }
    const value_type* address_data() const { return begin(); }
    
    
#if 0
    // Alleged ambiguity in MSVC 8.0, I need to turn off the warning 
    // For confusion with other vector assignments
    // For alleged ambiguity with scalar assign we omit template in CRTP
    // Removing the operator ends in run-time error
    vec_vec_asgn_expr<self, self> operator=( self const& e ) 
    {
	return vec_vec_asgn_expr<self, self>( *this, e );
    }
#endif

    /// Move assignment
    self& operator=(self src)
    {
	// Self-copy would be an indication of an error
	assert(this != &src);

	checked_change_dim(size(src));
	memory_base::move_assignment(src);
	return *this;
    }

#ifdef __PGI
    using crtp_base::operator=;
#else
    using assign_base::operator=;
#endif

    template <typename Value2> friend void fill(self&, const Value2&);

    /// swap two vectors
    friend void swap(self& vector1, self& vector2)
    {
	swap(static_cast<memory_base&>(vector1), static_cast<memory_base&>(vector2));
    }

    void change_dim(size_type n) { this->realloc(n); }
    void checked_change_dim(size_type n) { check_dim(n); change_dim(n); }
    
    void crop() {} // Only dummy here

} ; // dense_vector


// ================
// Free functions
// ================

template <typename Value, typename Parameters, typename Value2>
inline void fill(dense_vector<Value, Parameters>& vector, const Value2& value)
{
    std::fill(vector.begin(), vector.end(), value);    
}


template <typename Value, typename Parameters>
typename dense_vector<Value, Parameters>::size_type
inline num_rows_aux(const dense_vector<Value, Parameters>& , tag::row_major)
{
    return 1;
}

template <typename Value, typename Parameters>
typename dense_vector<Value, Parameters>::size_type
inline num_rows_aux(const dense_vector<Value, Parameters>& vector, tag::col_major)
{
    return size(vector);
}

/// Number of rows: is size for column vectors and 1 for row vectors
template <typename Value, typename Parameters>
typename dense_vector<Value, Parameters>::size_type
inline num_rows(const dense_vector<Value, Parameters>& vector)
{
    return num_rows_aux(vector, typename Parameters::orientation());
}


/// Number of columns: is size for row vectors and 1 for column vectors
template <typename Value, typename Parameters>
typename dense_vector<Value, Parameters>::size_type
inline num_cols(const dense_vector<Value, Parameters>& vector)
{
    return num_rows_aux(vector, typename mtl::traits::transposed_orientation<typename Parameters::orientation>::type());
}

/// Sub-vector function, more convenient with irange
template <typename Value, typename Parameters>
dense_vector<Value, Parameters>
inline sub_vector(dense_vector<Value, Parameters>& v, 
		  typename dense_vector<Value, Parameters>::size_type start,
		  typename dense_vector<Value, Parameters>::size_type finish)
{
    typedef dense_vector<Value, Parameters>    Vector;

    MTL_DEBUG_THROW_IF( is_negative(start) || is_negative(finish), index_out_of_range());
    irange r= intersection(irange(start, finish), irange(0, size(v)));
    return r.empty() ? Vector() : Vector(r.size(), &v[r.start()]);

#if 0
    finish= min(finish, size(v));
    start= min(start, finish); // implies min(start, size(v))
    return start < finish ? Vector(finish - start, &v[start]) : Vector();
#endif
}

template <typename Value, typename Parameters>
const dense_vector<Value, Parameters>
inline sub_vector(const dense_vector<Value, Parameters>& v, 
		  typename dense_vector<Value, Parameters>::size_type start,
		  typename dense_vector<Value, Parameters>::size_type finish)
{
    typedef dense_vector<Value, Parameters>    Vector;
    return sub_vector(const_cast<Vector&>(v), start, finish);
}


}} // namespace mtl::vector

namespace mtl {

    // Enable cloning of dense matrices
    template <typename Value, typename Parameters>
    struct is_clonable< vector::dense_vector<Value, Parameters> > : boost::mpl::true_ {};
        
} // namespace mtl

namespace mtl { namespace traits {


// ================
// Range generators
// For cursors
// ================

    template <typename Value, class Parameters>
    struct range_generator<tag::all, dense_vector<Value, Parameters> >
      : public detail::dense_element_range_generator<dense_vector<Value, Parameters>,
						     dense_el_cursor<Value>, complexity_classes::linear_cached>
    {};

    template <typename Value, class Parameters>
    struct range_generator<tag::nz, dense_vector<Value, Parameters> >
	: public range_generator<tag::all, dense_vector<Value, Parameters> >
    {};

    template <typename Value, class Parameters>
    struct range_generator<tag::iter::all, dense_vector<Value, Parameters> >
    {
	typedef dense_vector<Value, Parameters>   collection_t;
	typedef complexity_classes::linear_cached complexity;
	static int const                          level = 1;
	typedef typename collection_t::pointer    type;

	type begin(collection_t& collection)
	{
	    return collection.begin();
	}
	type end(collection_t& collection)
	{
	    return collection.end();
	}
    };

    template <typename Value, class Parameters>
    struct range_generator<tag::iter::nz, dense_vector<Value, Parameters> >
	: public range_generator<tag::iter::all, dense_vector<Value, Parameters> >
    {};

    template <typename Value, class Parameters>
    struct range_generator<tag::const_iter::all, dense_vector<Value, Parameters> >
    {
	typedef dense_vector<Value, Parameters>   collection_t;
	typedef complexity_classes::linear_cached complexity;
	static int const                          level = 1;
	typedef typename collection_t::const_pointer type;

	type begin(const collection_t& collection) const
	{
	    return collection.begin();
	}
	type end(const collection_t& collection) const
	{
	    return collection.end();
	}
    };

    template <typename Value, class Parameters>
    struct range_generator<tag::const_iter::nz, dense_vector<Value, Parameters> >
	: public range_generator<tag::const_iter::all, dense_vector<Value, Parameters> >
    {};

	
}} // namespace mtl::traits


#endif // MTL_DENSE_VECTOR_INCLUDE

