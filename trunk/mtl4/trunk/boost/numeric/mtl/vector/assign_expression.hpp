// $COPYRIGHT$

// Adapted from GLAS implementation by Karl Meerbergen and Toon Knappen


#ifndef MTL_VECTOR_ASSIGN_EXPRESSION_INCLUDE
#define MTL_VECTOR_ASSIGN_EXPRESSION_INCLUDE


namespace mtl { namespace vector {

// Model of VectorExpression
template <class E1, class E2>
class assign_expression 
{
public:
    // temporary solution
    typedef typename E1::value_type              value_type;
    // typedef typename glas::value< glas::scalar::assign_expression<typename E1::value_type, typename E2::value_type > >::type value_type ;
    
    // temporary solution
    typedef typename E1::size_type               size_type;
    //typedef typename utilities::smallest< typename E1::size_type, typename E2::size_type >::type                          size_type ;

    typedef value_type const_dereference_type ;

    typedef E1 first_argument_type ;
    typedef E2 second_argument_type ;
    

    assign_expression( first_argument_type& v1, second_argument_type const& v2 )
	: first( v1 ), second( v2 ), delayed_assign( false )
    {}

    ~assign_expression()
    {
	if (!delayed_assign)
	    for (size_type i= 0; i < first.size(); ++i)
		first( i )= second( i );
    }
    
    void delay_assign() { delayed_assign= true; }

    size_type size() const {
	assert( first.size() == second_.size() ) ;
	return first.size() ;
    }

     const_dereference_type operator() ( size_type i ) const {
	assert( delayed_assign );
        return first( i )= second( i ) ;
     }

     const_dereference_type operator[] ( size_type i ) const {
	assert( delayed_assign );
        return first( i )= second( i ) ;
     }

  private:
     first_argument_type&        first ;
     second_argument_type const& second ;
     bool                        delayed_assign;
  } ; // assign_expression

} } // Namespace mtl::vector


namespace mtl { namespace traits {

  template <class E1, class E2>
  struct category< vector::assign_expression<E1,E2> > {
    typedef vector_expr type ;
  } ;

}} // Namespace mtl::traits

#endif

