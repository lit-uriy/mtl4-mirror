// $COPYRIGHT$

// Adapted from GLAS implementation by Karl Meerbergen and Toon Knappen


#ifndef MTL_DENSE_VECTOR_INCLUDE
#define MTL_DENSE_VECTOR_INCLUDE


#include <iostream>
#include <cassert>
#include <vector>
#include <algorithm>
#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/utility/ashape.hpp>
#include <boost/numeric/mtl/utility/common_include.hpp>
#include <boost/numeric/mtl/vector/all_vec_expr.hpp>
#include <boost/numeric/mtl/vector/parameter.hpp>
#include <boost/numeric/mtl/detail/contiguous_memory_block.hpp>
#include <boost/numeric/mtl/utility/dense_el_cursor.hpp>
#include <boost/numeric/mtl/utility/range_generator.hpp>


namespace mtl { namespace vector {

template <class Value, typename Parameters = mtl::vector::parameters<> >
class dense_vector
    : public vec_expr<dense_vector<Value, Parameters> >,
      public detail::contiguous_memory_block< Value, Parameters::on_stack, Parameters::dimension::value >
{
    typedef detail::contiguous_memory_block< Value, Parameters::on_stack, Parameters::dimension::value >   super_memory;
public:
    typedef vec_expr<dense_vector<Value, Parameters> >  expr_base;
    typedef dense_vector      self;
    typedef Value             value_type ; 
    typedef std::size_t       size_type ;
    typedef value_type&       reference ;
    typedef value_type const& const_reference ;
    typedef Value*            pointer ;
    typedef Value const*      const_pointer ;
    typedef typename Parameters::orientation  orientation;

    typedef const_pointer     key_type;
    
    dense_vector( ) : expr_base( *this ), super_memory( Parameters::dimension::value ) {}
    
    dense_vector( size_type n )
	: expr_base( *this ), super_memory( n ) 
    {}
    
    dense_vector( size_type n, value_type value )
	: expr_base( *this ), super_memory( n ) 
    {
	std::fill(begin(), end(), value);
    }


    size_type size() const { return this->used_memory() ; }
    
    size_type stride() const { return 1 ; }

    void check_index( size_type i ) const
    {
	MTL_DEBUG_THROW_IF( i < 0 || i >= size(), bad_range());
    }

    reference operator()( size_type i ) 
    {
        check_index(i);
        return this->value_n( i ) ;
    }

    const_reference operator()( size_type i ) const 
    {
        check_index(i);
        return this->value_n( i ) ;
    }

    reference operator[]( size_type i ) 
    {
	return (*this)( i ) ;
    }

    const_reference operator[]( size_type i ) const 
    {
	return (*this)( i ) ;
    }

    void delay_assign() const {}

    const_pointer begin() const { return this->elements() ; }
    const_pointer end() const { return this->elements() + size() ; }
    
    pointer begin() { return this->elements() ; }
    pointer end() { return this->elements() + size() ; }

    vec_vec_asgn_expr<self, self> operator=( self const& e ) 
    {
	return vec_vec_asgn_expr<self, self>( *this, e );
    }

    template <class E>
    void check_consistent_shape( vec_expr<E> const& e ) const
    {
      #if 0 // leave this for later
	MTL_THROW_IF((!boost::is_same<
		         typename ashape::ashape<self>::type
		       , typename ashape::ashape<E>::type
		      >::value),
		     bad_range());
	// Might be optimized out by smart compilers
      #endif 
    }

    template <class E>
    vec_vec_asgn_expr<self, E> operator=( vec_expr<E> const& e )
    {
	check_consistent_shape(e);
	return vec_vec_asgn_expr<self, E>( *this, e.ref );
    }

    // Replace it later by expression (maybe)
    self& operator=(value_type value)
    {
	std::fill(begin(), end(), value);
	return *this;
    }

    template <class E>
    vec_vec_plus_asgn_expr<self, E> operator+=( vec_expr<E> const& e )
    {
	check_consistent_shape(e);
	return vec_vec_plus_asgn_expr<self, E>( *this, e.ref );
    }

    template <class E>
    vec_vec_minus_asgn_expr<self, E> operator-=( vec_expr<E> const& e )
    {
	check_consistent_shape(e);
	return vec_vec_minus_asgn_expr<self, E>( *this, e.ref );
    }

    friend std::ostream& operator<<( std::ostream& s, dense_vector<Value, Parameters> const& v ) 
    {
	s << "[" << v.size() << (traits::is_row_major<Parameters>::value ? "R" : "C") << "]{" ;
	for (size_type i=0; i < v.size()-1; ++ i) {
	    s << v(i) << "," ;
	}
	s << v(v.size()-1) << "}" ;
	return s ;
    }

} ; // dense_vector

}} // namespace mtl::vector



namespace mtl { namespace traits {

    template <typename Value, typename Parameters>
    struct is_row_major<dense_vector<Value, Parameters> >
	: public  is_row_major<Parameters>
    {};

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

	type begin(collection_t& collection) const
	{
	    return collection.begin();
	}
	type end(collection_t& collection) const
	{
	    return collection.begin();
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
	    return collection.begin();
	}
    };

    template <typename Value, class Parameters>
    struct range_generator<tag::const_iter::nz, dense_vector<Value, Parameters> >
	: public range_generator<tag::const_iter::all, dense_vector<Value, Parameters> >
    {};
	
}} // namespace mtl::traits


#endif // MTL_DENSE_VECTOR_INCLUDE

