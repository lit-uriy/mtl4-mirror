// $COPYRIGHT$

// Adapted from GLAS implementation by Karl Meerbergen and Toon Knappen


#ifndef MTL_VEC_VEC_ADD_EXPR_INCLUDE
#define MTL_VEC_VEC_ADD_EXPR_INCLUDE


namespace mtl { namespace vector {

// Model of VectorExpression
template <class E1, class E2>
class vec_vec_add_expr 
{
public:
    typedef vec_vec_add_expr                     self;

    // temporary solution
    typedef typename E1::value_type              value_type;
    // typedef typename glas::value< glas::scalar::vec_vec_add_expr<typename E1::value_type, typename E2::value_type > >::type value_type ;
    
    // temporary solution
    typedef typename E1::size_type               size_type;
    //typedef typename utilities::smallest< typename E1::size_type, typename E2::size_type >::type                          size_type ;

    typedef value_type                           const_dereference_type ;

    typedef E1                                   first_argument_type ;
    typedef E2                                   second_argument_type ;
    
public:
    vec_vec_add_expr( first_argument_type const& v1, second_argument_type const& v2 )
	: first( v1 ), second( v2 )
    {
	second.delay_assign();
    }
    
    void delay_assign() const {}

    size_type size() const
    {
	assert( first.size() == second.size() ) ;
	return first.size() ;
    }

    const_dereference_type operator() ( size_type i ) const
    {
        return first( i ) + second( i ) ;
    }

    const_dereference_type operator[] ( size_type i ) const
    {
        return first( i ) + second( i ) ;
    }

    template <typename Expr2>
    vec_vec_add_expr<self, Expr2> operator+ (const Expr2& expr2) const
    {
	return vec_vec_add_expr<self, Expr2>(*this, expr2);
    }

  private:
    first_argument_type const&  first ;
    second_argument_type const& second ;
  } ; // vec_vec_add_expr


} } // Namespace glas::vector





namespace mtl { namespace traits {

  template <class E1, class E2>
  struct category< vector::vec_vec_add_expr<E1,E2> > 
  {
      typedef tag::vector type ;
      // typedef tag::vector_expr type ;
  } ;

}} // Namespace mtl::traits

#endif

