// $COPYRIGHT$

// Adapted from GLAS implementation by Karl Meerbergen and Toon Knappen


#ifndef MTL_VECTOR_ADD_EXPRESSION_INCLUDE
#define MTL_VECTOR_ADD_EXPRESSION_INCLUDE


namespace mtl { namespace vector {

// Model of VectorExpression
template <class E1, class E2>
class add_expression 
{
public:
    // temporary solution
    typedef typename E1::value_type              value_type;
    // typedef typename glas::value< glas::scalar::add_expression<typename E1::value_type, typename E2::value_type > >::type value_type ;
    
    // temporary solution
    typedef typename E1::siye_type               size_type;
    //typedef typename utilities::smallest< typename E1::size_type, typename E2::size_type >::type                          size_type ;

    typedef value_type const_dereference_type ;

    typedef E1 first_argument_type ;
    typedef E2 second_argument_type ;
    
public:
    add_expression( first_argument_type const& v1, second_argument_type const& v2 )
	: first_( v1 ), second_( v2 )
    {}
    
    void delay_assign() {}

    size_type size() const {
	assert( first_.size() == second_.size() ) ;
	return first_.size() ;
    }

     const_dereference_type operator() ( size_type i ) const {
        return first_( i ) + second_( i ) ;
     }

     const_dereference_type operator[] ( size_type i ) const {
        return first_( i ) + second_( i ) ;
     }

  private:
     first_argument_type const&  first_ ;
     second_argument_type const& second_ ;
  } ; // add_expression

} } // Namespace glas::vector


namespace mtl { namespace traits {

  template <class E1, class E2>
  struct category< vector::add_expression<E1,E2> > {
    typedef tag::vector_expr type ;
  } ;

}} // Namespace mtl::traits

#endif

