#include <iostream>
#include <cmath>

#include <boost/numeric/linear_algebra/operators.hpp>
#include <boost/numeric/linear_algebra/new_concepts.hpp>
#include <boost/numeric/linear_algebra/power.hpp>

#if 0
namespace math {

    concept_map Monoid< mult<float>, float> {};
    concept_map Commutative< mult<float>, float> {};

    concept_map Group< add<float>, float> {};
}
 #endif

int main(int, char* []) 
{
    using math::power; using math::mult; using math::add;
    float a= 3.14;

    std::cout << "power(a, 2, mult<float>) " << power(a, 2, mult<float>()) << '\n';
    std::cout << "power(a, 2, add<float>) " << power(a, 2, add<float>()) << '\n';


    return 0;
}

