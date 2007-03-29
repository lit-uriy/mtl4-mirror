// $COPYRIGHT$

// Adapted from GLAS implementation by Karl Meerbergen and Toon Knappen


#ifndef MTL_DENSE_VECTOR_INCLUDE
#define MTL_DENSE_VECTOR_INCLUDE

//#include <boost/mpl/assert.hpp>
#include <iostream>
#include <cassert>
#include <vector>
#include <algorithm>
#include <boost/numeric/mtl/utility/common_include.hpp>
#include <boost/numeric/mtl/vector/vec_vec_add_expr.hpp>
#include <boost/numeric/mtl/vector/vec_vec_asgn_expr.hpp>

namespace mtl { namespace vector {

template <class T>
class dense_vector
{
public:
    typedef dense_vector      self;
    typedef T                 value_type ; // Should this be scalar_collection<T> or arithmethic_collection<T>
    typedef std::size_t       size_type ;
    typedef value_type&       reference ;
    typedef value_type const& const_reference ;
    typedef T*                pointer ;
    typedef T const*          const_pointer ;

    dense_vector( size_type n ): my_size( n ), data( new value_type[n] )  {}

    dense_vector( size_type n, value_type value ): my_size( n ), data( new value_type[n] )  
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

    template <typename E2>
    vec_vec_add_expr<self, E2> operator+ (const E2& e2) const
    {
	return vec_vec_add_expr<self, E2>(*this, e2);
    }

    const_pointer begin() const { return data ; }
    const_pointer end() const { return data+my_size ; }
    
    pointer begin() { return data ; }
    pointer end() { return data+my_size ; }

    vec_vec_asgn_expr<self, self> operator=( self const& e ) 
    {
	return vec_vec_asgn_expr<self, self>( *this, e );
    }


    template <class E>
    vec_vec_asgn_expr<self, E> operator=( E const& e )
    {
	return vec_vec_asgn_expr<self, E>( *this, e );
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

    //template <typename Value>



}} // Namespace mtl::vector

namespace mtl { 
    
    using vector::dense_vector;

    namespace traits {

	template <class T>
	struct category< dense_vector<T> > {
	    typedef tag::dense_vector   type ;
	} ;
    }

} // Namespace mtl::traits

#endif // MTL_DENSE_VECTOR_INCLUDE

