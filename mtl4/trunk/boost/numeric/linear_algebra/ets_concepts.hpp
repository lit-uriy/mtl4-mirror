// $COPYRIGHT$

#ifndef LA_ETS_CONCEPTS_INCLUDE
#define LA_ETS_CONCEPTS_INCLUDE

#include <boost/config/concept_macros.hpp>

#ifdef LA_WITH_CONCEPTS
#  include <concepts>
#else
#  warning "Concepts are not used"
#endif

#include <boost/numeric/linear_algebra/concepts.hpp>

//"Namespace ETS = Expression Templates Support":
// This file contains concepts that support the concepts defined in the namespace math.
// They are required since the math-concepts do not support ExpressionTemplates so far.
// The list of valid expressions is in fact infinite, so we just name some of them.
// Once ExpressionTemplates are supported by the math-concepts, the ets-concepts are no longer required.

namespace ets {

#ifdef LA_WITH_CONCEPTS

  auto concept Field<typename Element>
  {
    where std::Assignable<Element, math::AdditivePartiallyInvertibleMonoid<Element>::unary_result_type>; //"x=-y" valid
  };

  auto concept VectorSpace<typename Vector, typename Scalar>
  {
    // valid expression: "vector2 += scalar*vector1"
    typename res_type_1;
    res_type_1 operator+=(Vector&, math::Multiplicable<Scalar, Vector>::result_type);
    
    // valid expression: "vector2 -= scalar*vector1"
    typename res_type_2;
    res_type_2 operator-=(Vector&, math::Multiplicable<Scalar, Vector>::result_type);
    
    //valid expression: "vector *= -scalar"
    typename res_type_3;
    res_type_3 operator*=(Vector&, const math::AdditivePartiallyInvertibleMonoid<Scalar>::unary_result_type&);

    //valid expression: "vector3 = vector1 + scalar*vector2"
    where math::Addable<Vector, math::Multiplicable<Scalar, Vector>::result_type>; //"vector1+scalar*vector2" valid 
    where std::Assignable<Vector, math::Addable<Vector, math::Multiplicable<Scalar, Vector>::result_type>::result_type>; //"vector3 = vector1 + scalar*vector2" valid
  
  };
  

  auto concept InnerProduct<typename I, typename Vector, typename Scalar>
  {
    //valid expression: "scalar2 = scalar1/dot(vector1,vector2);"
    where math::Divisible<Scalar, std::Callable2<I, Vector, Vector>::result_type>;
    where std::Assignable<Scalar,  math::Divisible<Scalar, std::Callable2<I, Vector, Vector>::result_type>::result_type>;
  };

#endif//LA_WITH_CONCEPTS

} //namespace ets


#endif //LA_LAI_CONCEPTS_INCLUDE
