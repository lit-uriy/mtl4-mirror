#include <iostream>
#include <cmath>

#include <boost/numeric/linear_algebra/operators.hpp>
#include <boost/numeric/linear_algebra/concepts.hpp>

#include "algebraic_functions.hpp"


// User defined data types and operators


class positive_real 
{
    double value;
public:
    positive_real(double m): value(m) 
    {
	if (m < 0.0) throw "Negative value not allowed!\n"; 
    }

    double get_value() const 
    {
	return value; 
    }

    positive_real operator+(positive_real const& y) const
    {
	return value + y.value;
    }

    positive_real operator+=(positive_real const& y) const
    {
	return value + y.value;
    }    

    bool operator==(positive_real const& y) const
    {
	return value == y.value;
    }
    
    bool operator!=(positive_real const& y) const
    {
	return value != y.value;
    }
};


inline std::ostream& operator<< (std::ostream& stream, const positive_real& a) 
{
    return stream << a.get_value(); 
}

 
int main(int, char* [])  
{
    positive_real                a0(0.0), a2(2.0), a3(3.0), a4(4.0), a5(5.0);
    math::add<positive_real>     my_add;
  
    std::cout << "equal_results(a2,a5,  a3,a4, my_add) " 
	      << mtl::equal_results(a2,a5,  a3,a4, my_add)  << std::endl;
    std::cout << "equal_results(a2,a4,  a3,a4, my_add) " 
	      << mtl::equal_results(a2,a4,  a3,a4, my_add)  << std::endl;

    std::cout << "equal_add_results(a2,a5,  a3,a4) " 
	      << mtl::equal_add_results(a2,a5,  a3,a4)  << std::endl;
    std::cout << "equal_add_results(a2,a4,  a3,a4) " 
	      << mtl::equal_add_results(a2,a4,  a3,a4)  << std::endl;

    return 0;
}

