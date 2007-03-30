// $COPYRIGHT$

// Adapted from GLAS implementation by Karl Meerbergen and Toon Knappen


#ifndef MTL_VEC_VEC_ADD_EXPR_INCLUDE
#define MTL_VEC_VEC_ADD_EXPR_INCLUDE

#include <boost/numeric/mtl/vector/vec_expr.hpp>

namespace mtl { namespace vector {

// Model of VectorExpression
template <class E1, class E2>
class vec_vec_add_expr 
    : public vec_expr< vec_vec_add_expr<E1, E2> >
{
public:
    typedef vec_expr< vec_vec_add_expr<E1, E2> > expr_base;
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
	: expr_base( *this ), first( v1 ), second( v2 )
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

  private:
    first_argument_type const&  first ;
    second_argument_type const& second ;
} ; // vec_vec_add_expr

    
template <typename E1, typename E2>
inline vec_vec_add_expr<E1, E2>
operator+ (const vec_expr<E1>& e1, const vec_expr<E2>& e2)
{
    return vec_vec_add_expr<E1, E2>(e1.ref, e2.ref);
}


} } // Namespace mtl::vector





namespace mtl { namespace traits {

  template <class E1, class E2>
  struct category< vector::vec_vec_add_expr<E1,E2> > 
  {
      typedef tag::vector type ;
      // typedef tag::vector_expr type ;
  } ;

}} // Namespace mtl::traits

#endif

