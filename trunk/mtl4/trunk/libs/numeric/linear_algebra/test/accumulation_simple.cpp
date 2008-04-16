
#include <boost/numeric/linear_algebra/accumulate.hpp>
#include <boost/numeric/linear_algebra/concept_maps.hpp>
#include <boost/numeric/linear_algebra/operators.hpp>

#include <boost/timer.hpp>


template <typename Element>
void test_accumulate(const char* name)
{
    const int   array_size= 10;
    Element     array[array_size];
    for (int i= 0; i < array_size; i++) 
    	array[i]= Element(i);
    
    std::cout << '\n' << name << '\n' << " Add: ";
    math::accumulate(&array[0], array+array_size, Element(0), math::add<Element>());
    std::cout << "Mult: ";
    math::accumulate(array, array+array_size, Element(1), math::mult<Element>());
    std::cout << " Min: ";
    math::accumulate(array, array+array_size, Element(1), math::min<Element>());
    std::cout << " Max: ";
    math::accumulate(array, array+array_size, Element(1), math::max<Element>());
}


int main(int, char* [])
{
    test_accumulate<int>("int");
    test_accumulate<float>("float");
    test_accumulate<double>("double");

    return 0;
}
