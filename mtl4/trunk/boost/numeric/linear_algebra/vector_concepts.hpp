// $COPYRIGHT$

#ifndef LA_VECTOR_CONCEPTS_INCLUDE
#define LA_VECTOR_CONCEPTS_INCLUDE


#include <boost/numeric/linear_algebra/concepts.hpp>


#ifdef LA_WITH_CONCEPTS

namespace math {


concept VectorSpace<typename Vector, typename Scalar = Vector::value_type>
  where Field<Scalar>
{
    

}


}

#endif


#endif // LA_VECTOR_CONCEPTS_INCLUDE
