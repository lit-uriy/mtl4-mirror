// $COPYRIGHT$

#ifndef NEWSTD_COMPLEX_INCLUDE
#define NEWSTD_COMPLEX_INCLUDE

#include <concepts>
#include <cmath>
#include <sstream>
#include <iostream>

#include <boost/utility/enable_if.hpp>
#include <boost/mpl/bool.hpp>

// For clean defition of identity elements (instead of default constructor)
#include <boost/numeric/linear_algebra/identity.hpp>
// #include <boost/numeric/linear_algebra/new_concepts.hpp>

// enable_if is used here due to problems with constraining member functions
// conceptg++ also fails to constrain classes

namespace newstd {

    template <typename T> class complex;

    /// Concept for disambigation in template definitions
    concept IsComplex<typename T> 
    {
	typename value_type;
	value_type& T::real();
	const value_type& T::real() const;
	value_type& T::imag();
	const value_type& T::imag() const;
    }

    template <typename T> concept_map IsComplex< complex<T> > { typedef T value_type; }

    // Ugly hack, a shame to use it
    template <typename T> struct is_complex : boost::mpl::false_ {};
    template <typename T> struct is_complex<complex<T> > : boost::mpl::true_ {};

    /// @brief Describes types with a binary @c operator+
    auto concept HasPlusAssign<typename T, typename U = T>
    {
	typename result_type;
	result_type operator+=(T& t, U const& u);
	result_type operator+=(T& t, U&& u);
    }


/// Class for templates
template <typename T>
class complex
{
public:
    /// Value typedef.
    typedef T            value_type;
    typedef complex<T>   self;

    ///  Default constructor.  First parameter is x, second parameter is y.
    ///  Unspecified parameters default to 0.
    complex(const T& r = T(), const T & i = T()) : _real(r), _imag(i) {}
    //complex(const T& r = math::zero(T()), const T & i = math::zero(T())) : _real(r), _imag(i) {}

    template<typename U>
    complex(const U& s, boost::disable_if<is_complex<U> >) : _real(s), _imag(T()) {}

    template<typename U, typename V>
    complex(const U& r, const V& i) : _real(r), _imag(i) {}

    // Lets the compiler synthesize the copy constructor   
    // complex (const complex<T>&);

    // Killt conceptg++ if class is concept-restricted, i.e. seems to lead to infinit loop
    ///  Copy constructor.
    template<typename U>
    complex(const complex<U>& z) : _real(z.real()), _imag(z.imag()) {}

    ///  Return real part of complex number.
    T& real() { return _real; }
    ///  Return real part of complex number.
    const T& real() const { return _real; }
    ///  Return imaginary part of complex number.
    T& imag() { return _imag; }
    ///  Return imaginary part of complex number.
    const T& imag() const { return _imag; }


    /// Assign scalar @a s to this complex number.
    template <typename U> 
    // requires !IsComplex<U> && std::CopyAssignable<T, U> && std::DefaultConstructible<T> && std::CopyAssignable<T>
    typename boost::disable_if<is_complex<U>, self&>::type operator=(const U& s)
    {
	_real= s;
	_imag= T(); // math::zero(T());
	return *this;
    }

    /// Assign complex @a z to this complex number.
    template <typename U> // requires IsComplex<U> && std::CopyAssignable<T, typename IsComplex<U>::value_type> 
    typename boost::enable_if<is_complex<U>, self&>::type operator=(const U& z)
    {
	_real= z.real();
	_imag= z.imag();
	return *this;
    }


    /// Move-assign scalar @a s to this complex number.
    template <typename U> 
    // requires std::MoveAssignable<T, U> && !IsComplex<U> && std::DefaultConstructible<T> && std::CopyAssignable<T> self& 
    typename boost::disable_if<is_complex<U>, self&>::type operator=(U&& s)
    {
	_real= s;
	_imag= T(); // math::zero(T());
	return *this;
    }

    /// Move elements of complex @a z to elements of this complex number.
    template <typename U> 
    // requires std::MoveAssignable<T, U>  self& 
    typename boost::enable_if<is_complex<U>, self&>::type operator=(complex<U>&& z)
    {
	_real= real(z);
	_imag= imag(z);
	return *this;
    }

    // Move entire complex number
    // requires MoveAssignable<self>

    /// Add scalar @a t to this complex number.
    template <typename U> 
    // requires HasPlusAssign<T, U> && !IsComplex<U> self& 
    typename boost::disable_if<is_complex<U>, self&>::type operator+=(const U& s)
    {
	_real+= s;
	return *this;
    }
    
    /// Add complex @a z to this complex number.
    template <typename U> 
    // requires HasPlusAssign<T, U>  self& 
    typename boost::enable_if<is_complex<U>, self&>::type operator+=(const U& z)
    {
	_real+= real(z);
	_imag+= imag(z);
	return *this;
    }
    

    private:
      T _real;
      T _imag;


};



    ///  Return real part of complex number.
    template <typename T> 
    T& real(complex<T>& z) { return z.real(); }

    ///  Return real part of complex number.
    template <typename T> 
    const T& real(const complex<T>& z) { return z.real(); }
    
    ///  Return imaginary part of complex number.
    template <typename T> 
    T& imag(complex<T>& z) { return z.imag(); }
    
    ///  Return imaginary part of complex number.
    template <typename T> 
    const T& imag(const complex<T>& z) { return z.imag(); }


    /// Add scalar @a x with complex @a y
    // If U is assignable to the result's elements
    // not needed if compiler can optimize the zero away in previous version
    template <typename T, typename U>
    requires std::HasPlus<T, U> && !IsComplex<T> && std::CopyAssignable<T, U> 
    complex<std::HasPlus<T, U>::result_type> inline operator+(const T& x, const complex<U>& y)
    {
	return complex<std::HasPlus<T, U>::result_type>(x + real(y), imag(y));
    }

    /// Add complex @a x with scalar @a y
    // If T is assignable to the result's elements
    // not needed if compiler can optimize the zero away in previous version
    template <typename T, typename U>
    requires std::HasPlus<T, U> && !IsComplex<U> && std::CopyAssignable<T, U> 
    complex<std::HasPlus<T, U>::result_type> inline operator+(const complex<T>& x, const U& y)
    {
	return complex<std::HasPlus<T, U>::result_type>(real(x) + y, imag(x));
    }

    // General case if T is not assignable to result's elements
    /// Add complex @a x with complex @a y
    template <typename T, typename U>
    requires std::HasPlus<T, U>
    complex<std::HasPlus<T, U>::result_type> inline operator+(const complex<T>& x, const complex<U>& y)
    {
	return complex<std::HasPlus<T, U>::result_type>(real(x) + real(y), imag(x) + imag(y));
    }

    /// Compare scalar @a x with complex @a y
    template <typename T, typename U>
    requires !IsComplex<T> && std::EqualityComparable<T, U> && std::EqualityComparable<U> && std::DefaultConstructible<U>
    bool inline operator==(const T& x, const complex<U>& y)
    {
	return x == real(y) && imag(y) == U();
    }

    /// Compare complex @a x with scalar @a y
    template <typename T, typename U>
    requires !IsComplex<U> && std::EqualityComparable<T, U> && std::EqualityComparable<T> && std::DefaultConstructible<T>
    bool inline operator==(const complex<T>& x, const U& y)
    {
	return real(x) == y && imag(x) == T();
    }

    /// Compare complex @a x with complex @a y
    template <typename T, typename U>
    bool inline operator==(const complex<T>& x, const complex<U>& y)
    {
	return real(x) == real(y) && imag(x) == imag(y);
    }

    // Is there an elegant way with concepts to do this in one function?
    template <typename T, typename U>
    typename boost::enable_if_c<is_complex<T>::value || is_complex<U>::value, bool>::type
    inline operator!=(const T& x, const U& y)
    {
	return !(x == y);
    }


    ///  Insertion operator for complex values.
    template<typename _Tp, typename _CharT, class _Traits>
    std::basic_ostream<_CharT, _Traits>&
    operator<<(std::basic_ostream<_CharT, _Traits>& __os, const complex<_Tp>& __x)
    {
	std::basic_ostringstream<_CharT, _Traits> __s;
	__s.flags(__os.flags());
	__s.imbue(__os.getloc());
	__s.precision(__os.precision());
	__s << '(' << __x.real() << ',' << __x.imag() << ')';
	return __os << __s.str();
    }




// ==========
// Discussion
// ==========

#if 0
    // General case if U is not assignable to result's elements
    // Do we really need this???
    template <typename T, typename U>
    requires std::HasPlus<T, U> && !IsComplex<T>
    complex<std::HasPlus<T, U>::result_type> inline operator+(const T& x, const complex<U>& y)
    {
	return complex<std::HasPlus<T, U>::result_type>(x + real(y), math::zero(x) + imag(y));
    }
#endif



} // namespace newstd

#endif // NEWSTD_COMPLEX_INCLUDE
