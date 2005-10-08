#include <iostream>
#include <cmath>

#include <glas/operators.hpp>
#include "algebraic_functions.hpp"

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
	return myAge; 
    }

    age operator+(age y)
    {
	return my_age + y.my_age;
    }
    
    bool operator==(age y)
    {
	return my_age == y.my_age;
    }
};


inline std::ostream& operator<< (std::ostream& stream, const age& a) 
{
    return stream << a.sayAge(); 
}



int main(int, char* []) 
{
    age a0(0.0), a2(2.0), a3(3.0), a4(4.0), a5(5.0);
    glas::add<age>     ageAdd;
  
    std::cout << "equalResults(a2,a5,  a3,a4, ageAdd) " 
	      << equalResults(a2,a5,  a3,a4, ageAdd)  << std::endl;
    std::cout << "equalResults(a2,a4,  a3,a4, ageAdd) " 
	      << equalResults(a2,a4,  a3,a4, ageAdd)  << std::endl;
    std::cout << "identityPair(a2,a4, ageAdd) " 
	      << identityPair(a2,a4, ageAdd)  << std::endl;
    std::cout << "identityPair(a0,a0, ageAdd) " 
	      << identityPair(a0,a0, ageAdd)  << std::endl;
    return 0;
}

