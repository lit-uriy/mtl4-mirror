#include <iostream>
#include <cmath>

#include <glas/operators.hpp>
#include "algebraic_functions.hpp"
#include <boost/numeric/mtl/scalar/concepts.hpp>

// User defined data types and operators

class age 
{
    double my_age;
public:
    age(double m): my_age(m) 
    {
	if (m < 0.0) throw "Negative Age"; 
    }

    double sayAge() const 
    {
	return my_age; 
    }

    age operator+(age const y) const
    {
	return my_age + y.my_age;
    }

    bool operator==(age const& y) const
    {
	return my_age == y.my_age;
    }
    
    bool operator!=(age const& y) const
    {
	return my_age != y.my_age;
    }

};
 

namespace mtl { 
  template<> concept Monoid <age, glas::add<age> > {};
}

inline std::ostream& operator<< (std::ostream& stream, const age& a) 
{
    return stream << a.sayAge(); 
}



int main(int, char* []) 
{
    age a0(0.0), a2(2.0), a3(3.0), a4(4.0), a5(5.0);
    glas::add<age>     ageAdd;
  
    std::cout << "equal_results(a2,a5,  a3,a4, ageAdd) " 
	      << mtl::equal_results(a2,a5,  a3,a4, ageAdd)  << '\n';
    std::cout << "equal_results(a2,a4,  a3,a4, ageAdd) " 
	      << mtl::equal_results(a2,a4,  a3,a4, ageAdd)  << '\n';

    std::cout << "identity_pair(a2,a4, ageAdd) " 
	      << mtl::identity_pair(a2,a4, ageAdd)  << '\n';
    std::cout << "identity_pair(a0,a0, ageAdd) " 
	      << mtl::identity_pair(a0,a0, ageAdd)  << '\n';

    return 0;
}

