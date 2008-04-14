#include <iostream>
#include <cmath>

#include <boost/numeric/linear_algebra/operators.hpp>
#include <boost/numeric/linear_algebra/inverse.hpp>
#include <boost/numeric/linear_algebra/is_invertible.hpp>
#include <boost/numeric/linear_algebra/new_concepts.hpp>
// #include <boost/numeric/linear_algebra/concept_maps.hpp>
#include <boost/numeric/linear_algebra/power.hpp>

namespace math {

#if 0
    concept_map Monoid< mult<float>, float> {};
    concept_map Commutative< mult<float>, float> {};

    concept_map Group< add<float>, float> {};
 #endif

    concept_map SignedIntegral<int> {}

    concept_map AbelianGroup< add<float>, float > {}
    //concept_map PIMonoid< add<float>, float > {}
    //concept_map Monoid< add<float>, float > {}
    //concept_map SemiGroup< add<float>, float > {}

    concept_map SemiGroup< mult<float>, float > {}
}

int main(int, char* []) 
{
    using math::power; using math::mult; using math::add;
    float a= 3.14;

    std::cout << "power(a, 5, mult<float>) " << power(a, 5, mult<float>()) << '\n';
    std::cout << "power(a, 5, add<float>) " << power(a, 5, add<float>()) << '\n';


    return 0;
}

