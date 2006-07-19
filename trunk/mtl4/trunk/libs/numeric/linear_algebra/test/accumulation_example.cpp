#include <iostream>

#include <boost/numeric/linear_algebra/operators.hpp>
#include <boost/numeric/linear_algebra/concepts.hpp>

#include "algebraic_functions.hpp"

// float has a concept map for Field 
// and PartiallyInvertibleMonoid w.r.t. mult
// This implies concept maps for all other scalar concepts

using math::identity; using math::add;

int main(int, char* [])
{
    using namespace std;
    using namespace mtl;

    math::add<float>    float_add;
    math::mult<float>   float_mult;

    const int   array_size= 2000;
    float       array[array_size];
    
    for (int i= 0; i < array_size; i++)
	array[i]= (float)(i);

    cout << "STL accumulate" << std::accumulate(array, array+array_size, 0.0, float_add) << '\n';
    
    // cout << "Unrolled accumulate" << accumulate_unrolled(array, array+array_size, 0.0, float_add) << '\n';

   return 0;
}
