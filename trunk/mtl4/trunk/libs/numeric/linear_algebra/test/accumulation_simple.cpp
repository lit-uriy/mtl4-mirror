
#include <boost/numeric/linear_algebra/accumulate.hpp>
#include <boost/numeric/linear_algebra/concept_maps.hpp>
#include <boost/numeric/linear_algebra/operators.hpp>

#include <boost/timer.hpp>


template <typename Element>
void test_accumulate(const char* name)
{
    using math::accumulate; using std::cout;

    const int   array_size= 10;
    Element     array[array_size];
    for (int i= 0; i < array_size; i++) 
    	array[i]= Element(i);
    
    std::cout << '\n' << name << '\n' << " Add: ";
    mtl::accumulate(&array[0], array+array_size, Element(0), math::add<Element>());
    std::cout << "Mult: ";
    mtl::accumulate(array, array+array_size, Element(1), math::mult<Element>());
    std::cout << " Min: ";
    mtl::accumulate(array, array+array_size, Element(1), math::min<Element>());
    std::cout << " Max: ";
    mtl::accumulate(array, array+array_size, Element(1), math::max<Element>());
}


int main(int, char* [])
{
    test_accumulate<int>("int");
    test_accumulate<float>("float");
    test_accumulate<double>("double");

    return 0;
}
