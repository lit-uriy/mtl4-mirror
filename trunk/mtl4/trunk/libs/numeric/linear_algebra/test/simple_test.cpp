#include <iostream>
#include <cmath>


#ifdef __GXX_CONCEPTS__
#  include <concepts>
#else 
#  include <boost/numeric/linear_algebra/pseudo_concept.hpp>
#endif


namespace algebra {

    concept Commutative<typename Operation, typename Element>
    {
	axiom Commutativity(Operation op, Element x, Element y)
	{
	    op(x, y) == op(y, x); 
	}   
    };

    concept Associative<typename Operation, typename Element>
    {
        axiom Associativity(Operation op, Element x, Element y, Element z)
        {
	    op(x, op(y, z)) == op(op(x, y), z); 
        }
    };


    auto concept SemiGroup<typename Operation, typename Element>
      : Associative<Operation, Element>
    {};



}

namespace math {

// ==================================
// Classification of Arithmetic Types
// ==================================

// In addtion to std::Integral
concept Float<typename T> 
  : std::DefaultConstructible<T>, std::CopyConstructible<T>,
    std::LessThanComparable<T>, std::EqualityComparable<T>
{
  T operator+(T);
  T operator+(T, T);
  T& operator+=(T&, T);
  T operator-(T, T);
  T operator-(T);
  T& operator-=(T&, T);
  T operator*(T, T);
  T& operator*=(T&, T);
  T operator/(T, T);
  T& operator/=(T&, T);

  // TBD: Some day, these will come from LessThanComparable,
  // EqualityComparable, etc.
  bool operator>(T, T);
  bool operator<=(T, T);
  bool operator>=(T, T);
  bool operator!=(T, T);

  //requires std::Assignable<T>, std::SameType<std::Assignable<T>::result_type, T&>;
}

concept_map Float<float> {}
concept_map Float<double> {}
concept_map Float<long double> {}

// The difference to Float is the lack of LessThanComparable
concept Complex<typename T> 
  : std::DefaultConstructible<T>, std::CopyConstructible<T>,
    std::EqualityComparable<T>
{
  T operator+(T);
  T operator+(T, T);
  T& operator+=(T&, T);
  T operator-(T, T);
  T operator-(T);
  T& operator-=(T&, T);
  T operator*(T, T);
  T& operator*=(T&, T);
  T operator/(T, T);
  T& operator/=(T&, T);

  // TBD: Some day, these will come from EqualityComparable
  bool operator!=(T, T);

  //requires std::Assignable<T>, std::SameType<std::Assignable<T>::result_type, T&>;
}

template <typename T>
  requires Float<T>
concept_map Complex<std::complex<T> > {}

} // namespace math


int main(int, char* [])  
{

    return 0;
}
