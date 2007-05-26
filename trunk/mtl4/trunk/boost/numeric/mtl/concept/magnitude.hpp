// $COPYRIGHT$

#ifndef MTL_MAGNITUDE_INCLUDE
#define MTL_MAGNITUDE_INCLUDE

#include <complex>

namespace mtl {

// =================================================
// Concept to specify return type of abs (and norms)
// =================================================


#ifdef __GXX_CONCEPTS__

// Concept to specify to specify projection of scalar value to comparable type
// For instance as return type of abs
// Minimalist definition for maximal applicability
auto concept Magnitude<typename T>
{
    typename type = T;
};

template <typename T>
concept_map Magnitude<std::complex<T> >
{
    typedef T type;
}


// Concept for norms etc., which are real values in mathematical definitions
auto concept RealMagnitude<typename T>
  : Magnitude<T>
{
    requires std::EqualityComparable<type>;
    requires std::LessThanComparable<type>;

    requires Field<type>;

    type sqrt(type);

    type abs(T);
}

#else  // now without concepts

template <typename T>
struct Magnitude
{
    typedef T type;
};

template <typename T>
struct Magnitude<std::complex<T> >
{
    typedef T type;
};

template <typename T> struct RealMagnitude
  : public Magnitude<T>
{};

#endif  // __GXX_CONCEPTS__

} // namespace mtl

#endif // MTL_MAGNITUDE_INCLUDE
