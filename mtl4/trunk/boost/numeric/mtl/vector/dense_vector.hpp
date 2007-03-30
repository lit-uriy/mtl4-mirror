// $COPYRIGHT$

// Adapted from GLAS implementation by Karl Meerbergen and Toon Knappen


#ifndef MTL_DENSE_VECTOR_INCLUDE
#define MTL_DENSE_VECTOR_INCLUDE


#include <iostream>
#include <cassert>
#include <vector>
#include <algorithm>
#include <boost/numeric/mtl/utility/common_include.hpp>
#include <boost/numeric/mtl/vector/vec_expr.hpp>
#include <boost/numeric/mtl/vector/vec_vec_plus_expr.hpp>
#include <boost/numeric/mtl/vector/vec_vec_minus_expr.hpp>
#include <boost/numeric/mtl/vector/vec_vec_asgn_expr.hpp>
#include <boost/numeric/mtl/vector/parameter.hpp>


namespace mtl { namespace vector {

template <class T, typename Parameters = mtl::vector::parameters<> >
class dense_vector
    : public vec_expr<dense_vector<T> >
{
public:
    typedef vec_expr<dense_vector<T> >  expr_base;
    typedef dense_vector      self;
    typedef T                 value_type ; 
    typedef std::size_t       size_type ;
    typedef value_type&       reference ;
    typedef value_type const& const_reference ;
    typedef T*                pointer ;
    typedef T const*          const_pointer ;
    typedef typename Parameters::orientation  orientation;

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

    size_type size() const { return my_size ; }
    
    size_type stride() const { return 1 ; }

    reference operator()( size_type i ) 
    {
        assert( i<my_size ) ;
        return data[ i ] ;
    }

    const_reference operator()( size_type i ) const 
    {
        assert( i<my_size ) ;
        return data[ i ] ;
    }

    reference operator[]( size_type i ) 
    {
        assert( i<my_size ) ;
        return data[ i ] ;
    }

    const_reference operator[]( size_type i ) const 
    {
        assert( i<my_size ) ;
        return data[ i ] ;
    }

    void delay_assign() const {}

    const_pointer begin() const { return data ; }
    const_pointer end() const { return data+my_size ; }
    
    pointer begin() { return data ; }
    pointer end() { return data+my_size ; }

    vec_vec_asgn_expr<self, self> operator=( self const& e ) 
    {
	return vec_vec_asgn_expr<self, self>( *this, e );
    }


    template <class E>
    vec_vec_asgn_expr<self, E> operator=( vec_expr<E> const& e )
    {
	return vec_vec_asgn_expr<self, E>( *this, e.ref );
    }

    // Replace it later by expression (maybe)
    self& operator=(value_type value)
    {
	std::fill(data, data+my_size, value);
	return *this;
    }


    friend std::ostream& operator<<( std::ostream& s, dense_vector<T> const& v ) 
    {
	s << "[" << v.my_size << "]{" ;
	for (size_type i=0; i < v.my_size-1; ++ i) {
	    s << v(i) << "," ;
	}
	s << v(v.my_size-1) << "}" ;
	return s ;
    }

  private:
     size_type my_size ;
     pointer   data ;
} ; // dense_vector


}} // Namespace mtl::vector

namespace mtl { 
    
    // using vector::dense_vector;

}

#endif // MTL_DENSE_VECTOR_INCLUDE

