// $COPYRIGHT$

// Adapted from GLAS implementation by Karl Meerbergen and Toon Knappen


#ifndef MTL_VEC_VEC_AOP_EXPR_INCLUDE
#define MTL_VEC_VEC_AOP_EXPR_INCLUDE

#include <boost/numeric/mtl/vector/vec_expr.hpp>
#include <boost/numeric/mtl/utility/sfunctor.hpp>


namespace mtl { namespace vector {

// Generic assign operation expression template for vectors
// Model of VectorExpression
template <class E1, class E2, typename SFunctor>
class vec_vec_aop_expr 
    : public vec_expr< vec_vec_aop_expr<E1, E2, SFunctor> >
{
public:
    typedef vec_expr< vec_vec_aop_expr<E1, E2, SFunctor> >  expr_base;
    typedef typename E1::value_type              value_type;
    
    // temporary solution
    typedef typename E1::size_type               size_type;
    //typedef typename utilities::smallest< typename E1::size_type, typename E2::size_type >::type                          size_type ;

    typedef value_type reference_type ;

    typedef E1 first_argument_type ;
    typedef E2 second_argument_type ;
    
    vec_vec_aop_expr( first_argument_type& v1, second_argument_type const& v2 )
	: expr_base( *this ), first( v1 ), second( v2 ), delayed_assign( false )
    {
	second.delay_assign();
    }

    ~vec_vec_aop_expr()
    {
	if (!delayed_assign)
	    for (size_type i= 0; i < first.size(); ++i)
		SFunctor::apply( first(i), second(i) );
    }
    
    void delay_assign() const { delayed_assign= true; }

    size_type size() const {
	assert( first.size() == second.size() ) ;
	return first.size() ;
    }

     value_type& operator() ( size_type i ) const {
	assert( delayed_assign );
	return SFunctor::apply( first(i), second(i) );
     }

     value_type& operator[] ( size_type i ) const{
	assert( delayed_assign );
	return SFunctor::apply( first(i), second(i) );
     }

  private:
     mutable first_argument_type&        first ;
     second_argument_type const&         second ;
     mutable bool                        delayed_assign;
  } ; // vec_vec_aop_expr

} } // Namespace mtl::vector





#endif

