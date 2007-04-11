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

#include <boost/numeric/mtl/utility/ashape.hpp>
#include <boost/numeric/mtl/utility/common_include.hpp>
#include <boost/numeric/mtl/vector/all_vec_expr.hpp>
#include <boost/numeric/mtl/vector/parameter.hpp>


namespace mtl { namespace vector {

template <class Value, typename Parameters = mtl::vector::parameters<> >
class dense_vector
    : public vec_expr<dense_vector<Value> >,
      public detail::contiguous_memory_matrix< Value, Parameters::on_stack, Parameters::dimension::value >
{
    typedef detail::contiguous_memory_matrix< Value, Parameters::on_stack, Parameters::dimension::value >   super_memory;
public:
    typedef vec_expr<dense_vector<Value> >  expr_base;
    typedef dense_vector      self;
    typedef Value             value_type ; 
    typedef std::size_t       size_type ;
    typedef value_type&       reference ;
    typedef value_type const& const_reference ;
    typedef Value*            pointer ;
    typedef Value const*      const_pointer ;
    typedef typename Parameters::orientation  orientation;

    
    







#if 0
    dense_vector( size_type n )
	: expr_base( *this ), my_size( n ), data( new value_type[n] )  
    {}

    dense_vector( size_type n, value_type value )
	: expr_base( *this ), my_size( n ), data( new value_type[n] )  
    {
	std::fill(data, data+my_size, value);
    }

    ~dense_vector() {
        delete[] data ;
    }
#endif

    size_type size() const { return this->used_memory() ; }
    
    size_type stride() const { return 1 ; }

    void check_index( size_type i )
    {
	MTL_DEBUG_THROW_IF( i < 0 || i > size(), bad_range());
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
    vec_vec_asgn_expr<self, E> operator=( vec_expr<E> const& e )
    {
#if 0
	BOOST_STATIC_ASSERT((boost::is_same<typename ashape::ashape<self>::type, 
			                    typename ashape::ashape<E>::type>::value));
#endif
	return vec_vec_asgn_expr<self, E>( *this, e.ref );
    }

    // Replace it later by expression (maybe)
    self& operator=(value_type value)
    {
	std::fill(data, data+my_size, value);
	return *this;
    }

    template <class E>
    vec_vec_plus_asgn_expr<self, E> operator+=( vec_expr<E> const& e )
    {
#if 0
	BOOST_STATIC_ASSERT((boost::is_same<typename ashape::ashape<self>::type, 
			                    typename ashape::ashape<E>::type>::value));
#endif
	return vec_vec_plus_asgn_expr<self, E>( *this, e.ref );
    }

    template <class E>
    vec_vec_minus_asgn_expr<self, E> operator-=( vec_expr<E> const& e )
    {
#if 0
	BOOST_STATIC_ASSERT((boost::is_same<typename ashape::ashape<self>::type, 
			                    typename ashape::ashape<E>::type>::value));
#endif
	return vec_vec_minus_asgn_expr<self, E>( *this, e.ref );
    }

    friend std::ostream& operator<<( std::ostream& s, dense_vector<Value> const& v ) 
    {
	s << "[" << v.my_size << "]{" ;
	for (size_type i=0; i < v.my_size-1; ++ i) {
	    s << v(i) << "," ;
	}
	s << v(v.my_size-1) << "}" ;
	return s ;
    }

  private:
    // size_type my_size ;
    // pointer   data ;
} ; // dense_vector


}} // Namespace mtl::vector


#endif // MTL_DENSE_VECTOR_INCLUDE

