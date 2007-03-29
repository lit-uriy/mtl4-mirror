// $COPYRIGHT$

// Adapted from GLAS implementation by Karl Meerbergen and Toon Knappen


#ifndef MTL_VEC_VEC_ASGN_EXPR_INCLUDE
#define MTL_VEC_VEC_ASGN_EXPR_INCLUDE


namespace mtl { namespace vector {

// Model of VectorExpression
template <class E1, class E2>
class vec_vec_asgn_expr 
{
public:
    // temporary solution
    typedef typename E1::value_type              value_type;
    // typedef typename glas::value< glas::scalar::vec_vec_asgn_expr<typename E1::value_type, typename E2::value_type > >::type value_type ;
    
    // temporary solution
    typedef typename E1::size_type               size_type;
    //typedef typename utilities::smallest< typename E1::size_type, typename E2::size_type >::type                          size_type ;

    typedef value_type reference_type ;

    typedef E1 first_argument_type ;
    typedef E2 second_argument_type ;
    

    vec_vec_asgn_expr( first_argument_type& v1, second_argument_type const& v2 )
	: first( v1 ), second( v2 ), delayed_assign( false )
    {
	second.delay_assign();
    }

    ~vec_vec_asgn_expr()
    {
	if (!delayed_assign)
	    for (size_type i= 0; i < first.size(); ++i)
		first( i )= second( i );
    }
    
    void delay_assign() const { delayed_assign= true; }

    size_type size() const {
	assert( first.size() == second.size() ) ;
	return first.size() ;
    }

     value_type operator() ( size_type i ) const {
	assert( delayed_assign );
        return first( i )= second( i ) ;
     }

     value_type operator[] ( size_type i ) const{
	assert( delayed_assign );
        return first( i )= second( i ) ;
     }

  private:
     mutable first_argument_type&        first ;
     second_argument_type const& second ;
     mutable bool                delayed_assign;
  } ; // vec_vec_asgn_expr

} } // Namespace mtl::vector


namespace mtl { namespace traits {

  template <class E1, class E2>
  struct category< vector::vec_vec_asgn_expr<E1,E2> > 
  {
      typedef tag::vector type ;
  } ;

}} // Namespace mtl::traits

#endif

