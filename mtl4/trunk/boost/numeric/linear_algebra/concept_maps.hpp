// $COPYRIGHT$

#ifndef MATH_CONCEPT_MAPS_INCLUDE
#define MATH_CONCEPT_MAPS_INCLUDE

#include <boost/numeric/linear_algebra/intrinsic_concept_maps.hpp>
#include <boost/numeric/linear_algebra/new_concepts.hpp>


namespace math {

#if 0 // Why this doesn't work???
    template <typename T>
        requires IntrinsicUnsignedIntegral<T>
    concept_map UnsignedIntegral<T> {}

    template <typename T>
        requires IntrinsicSignedIntegral<T>
    concept_map SignedIntegral<T> {}
#endif
    
    concept_map UnsignedIntegral<unsigned int> {}

    concept_map SignedIntegral<int> {}

    concept_map AbelianGroup< add<float>, float > {}
    concept_map SemiGroup< mult<float>, float > {}



} // namespace math

#endif // MATH_CONCEPT_MAPS_INCLUDE
