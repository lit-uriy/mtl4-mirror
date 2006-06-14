#include <iostream>

#include <boost/operators.hpp>
#include <boost/numeric/linear_algebra/identity.hpp>
#include <boost/numeric/linear_algebra/is_invertible.hpp>
#include <boost/numeric/linear_algebra/inverse.hpp>
#include <boost/numeric/linear_algebra/operators.hpp>
#include <boost/numeric/linear_algebra/concepts.hpp>
#include <boost/numeric/meta_math/is_prime.hpp>
#include <cassert>

#include "algebraic_functions.hpp"

using std::ostream;


template<typename T, T N>
// where Integral<T>
class mod_n_t 
  : boost::equality_comparable< mod_n_t<T, N> >,
    boost::less_than_comparable< mod_n_t<T, N> >,
    boost::addable< mod_n_t<T, N> >,
    boost::subtractable< mod_n_t<T, N> >,
    boost::multipliable< mod_n_t<T, N> >,
    boost::dividable< mod_n_t<T, N> >
{
    // or BOOST_STATIC_ASSERT((IS_INTEGRAL))

    T                 value;
 public:
    typedef T         value_type;
    typedef mod_n_t   self;

    static const T modulo= N;

    explicit mod_n_t(T const& v) : value( v % modulo ) {}

    // modulo of negative numbers can be bizarre
    // better use constructor for T
    explicit mod_n_t(int v) 
    {
	value= v >= 0 ? v%modulo : modulo - -v%modulo; 
    }

    // copy constructor
    mod_n_t(const mod_n_t<T, N>& m): value(m.get()) {}
  
    mod_n_t<T, N>& operator= (const mod_n_t<T, N>& m) 
    {
	value= m.value; 
	return *this; 
    }
    
    // conversion from other moduli must be called explicitly
    template<T OtherN>
    mod_n_t<T, N>& convert(const mod_n_t<T, OtherN>& m) 
    {
	value= m.value >= 0 ? m.value%modulo : modulo - -m.value%modulo; 
	return *this; 
    }

    T get() const 
    {
	return value; 
    }

    bool operator==(self const& y) const
    {
	check(*this); check(y);
	return this->value == y.value;
    }

    bool operator<(self const& y) const
    {
	check(*this); check(y);
	return this->value < y.value;
    }

    self operator+= (self const& y)
    {
	check(*this); check(y);
	this->value += y.value;
	this->value %= modulo;
	return *this;
    }

    self operator-= (self const& y)
    {
	check(*this); check(y);
	// add n to avoid negative numbers esp. if T is unsigned
	this->value += modulo;
	this->value -= y.value;
	this->value %= modulo;
	return *this;
    }

    self operator*= (self const& y)
    {
	check(*this); check(y);
	this->value *= y.value;
	this->value %= modulo;
	return *this;
    }

    self operator/= (self const& y);
    
};

template<typename T, T N>
inline void check(const mod_n_t<T, N>& x)
{
    assert(x.get() >= 0 && x.get() < N);
}

template<typename T, T N>
inline ostream& operator<< (ostream& stream, const mod_n_t<T, N>& a) 
{
    check(a);
    return stream << a.get(); 
}


// Extended Euclidian algorithm in vector notation
    // uu = (u1, u2, u3) := (1, 0, u)
    // vv = (v1, v2, v3) := (0, 1, v)
    // while (u3 % v3 != 0) {
    //   q= u3 / v3
    //   rr= uu - q * vv
    //   uu= vv
    //   vv= rr }
    // 
    // with u = N and v = y
    // --> v2 * v mod u == gcd(u, v)
    // --> v2 * y mod N == 1
    // --> x * v2 == x / y
    // v1, u1, and r1 not used
template<typename T, T N>
inline mod_n_t<T, N> mod_n_t<T, N>::operator/= (const mod_n_t<T, N>& y) 
{
    check(*this); check(y);
    if (y.get() == 0) throw "Division by 0";

    // Goes wrong with unsigned b/c some values will be negative (even if the result isn't)
    // Something like remove_sign<T>::type would be cute
    int u= N, v= y.get(), /* u1= 1, */  u2= 0, /* v1= 0, */  v2= 1, q, r, /* r1, */  r2;

    while (u % v != 0) {
	q= u / v;

	r= u % v; /* r1= u1 - q * v1; */ r2= u2 - q * v2;
	u= v; /* u1= v1; */ u2= v2;
	v= r; /* v1= r1; */ v2= r2;
    }

    return *this *= mod_n_t<T, N>(v2); 
}

inline int gcd(int u, int v)
{
    int r;
    while ((r= u % v) != 0) {
	u= v; v= r;
    }
    return v;
}

namespace math {

    template<typename T, T N>
    struct identity_t< add< mod_n_t<T, N> >, mod_n_t<T, N> >
    {
	typedef mod_n_t<T, N> mod_t;
	mod_t operator() (add<mod_t> const&, mod_t const& v) const
	{
	    return mod_t(0);
	}
    };


    // Reverse definition, a little more efficient if / uses inverse
    template<typename T, T N>
    struct inverse_t< add< mod_n_t<T, N> >, mod_n_t<T, N> >
    {
	typedef mod_n_t<T, N> mod_t;
	mod_t operator() (add<mod_t> const& op, mod_t const& v) const
	{
	    return identity(op, v) - v;
	}
    };
    

    template<typename T, T N>
    struct is_invertible_t< add< mod_n_t<T, N> >, mod_n_t<T, N> >
    {
	typedef mod_n_t<T, N> mod_t;
	bool operator() (add<mod_t> const&, mod_t const& v) const
	{ return true; }
    };
    

    template<typename T, T N>
    struct identity_t< mult< mod_n_t<T, N> >, mod_n_t<T, N> >
    {
	typedef mod_n_t<T, N> mod_t;
	mod_t operator() (mult<mod_t> const&, mod_t const& v) const
	{
	    return mod_t(1);
	}
    };


    // Reverse definition, a little more efficient if / uses inverse
    template<typename T, T N>
    struct inverse_t< mult< mod_n_t<T, N> >, mod_n_t<T, N> >
    {
	typedef mod_n_t<T, N> mod_t;
	mod_t operator() (mult<mod_t> const&, mod_t const& v) const
	{
	    return mod_t(1) / v;
	}
    };
    

    template<typename T, T N>
    struct is_invertible_t< mult< mod_n_t<T, N> >, mod_n_t<T, N> >
    {
	typedef mod_n_t<T, N> mod_t;
	bool operator() (mult<mod_t> const&, mod_t const& v) const
	{
	    T value = v.get();
	    return value != 0 && gcd(N, value) == 1;
	}
    };
    

# ifdef LA_WITH_CONCEPTS

// Concept mapping
// All modulo sets are commutative rings with identity
// but only if N is prime it is also a field
// Due to some mapping nesting trouble we define normally derived maps


template <typename T, T N>
concept_map AbelianGroup< add< mod_n_t<T, N> >, mod_n_t<T, N> > 
{
    // Why do we need the typedefs???
    
    typedef mod_n_t<T, N> inverse_result_type;
    typedef mod_n_t<T, N> identity_result_type;
    typedef bool          is_invertible_result_type;
}

template <typename T, T N>
concept_map PartiallyInvertibleCommutativeMonoid< mult< mod_n_t<T, N> >, mod_n_t<T, N> > 
{
    // Why do we need the typedefs???
    
    typedef mod_n_t<T, N> inverse_result_type;
    typedef mod_n_t<T, N> identity_result_type;
    typedef bool          is_invertible_result_type;
}

template <typename T, T N>
concept_map GenericCommutativeRingWithIdentity
       < add< mod_n_t<T, N> >, 
	 mult< mod_n_t<T, N> >, 
	 mod_n_t<T, N> 
       > {}

#if 0 
template <typename T, T N>
    where meta_math::Prime<N>
concept_map CommutativeSemiGroup< mult< mod_n_t<T, N> >, mod_n_t<T, N> > {} 
#endif

template <typename T, T N>
    where meta_math::Prime<N>
concept_map GenericField
       < add< mod_n_t<T, N> >, 
	 mult< mod_n_t<T, N> >, 
	 mod_n_t<T, N> 
       > 
{
    // Why do we need the typedefs???
    
    typedef mod_n_t<T, N> inverse_result_type;
    typedef mod_n_t<T, N> identity_result_type;
    typedef bool          is_invertible_result_type;
}


template <typename T, T N>
concept_map CommutativeRingWithIdentity< mod_n_t<T, N> > 
{
    // Why do we need the typedefs???
    
    typedef mod_n_t<T, N>& assign_result_type;
    typedef mod_n_t<T, N>  result_type;
    typedef mod_n_t<T, N>  unary_result_type;
}


template <typename T, T N>
    where meta_math::Prime<N>
concept_map Field< mod_n_t<T, N> >
{
    // Why do we need the typedefs???
    
    typedef mod_n_t<T, N>& assign_result_type;
    typedef mod_n_t<T, N>  result_type;
    typedef mod_n_t<T, N>  unary_result_type;
}


// Shall be only called if T is a Field 
template <typename T>
   where Field<T>
bool is_field(const T&)
{
    return true;
}

// Shall be only called if T is not a Field 
template <typename T>
bool is_field(const T&)
{
    return false;
}

#if 0
// Shall be only called if T is a Field 
template <typename Add, typename Mult, typename T>
   where GenericField<Add, Mult, T>
bool is_field(const Add&, const Mult&, const T&)
{
    return true;
}

// Shall be only called if T is not a Field 
template <typename Add, typename Mult, typename T>
bool is_field(const Add&, const Mult&, const T&)
{
    return false;
}
#endif

# endif // LA_WITH_CONCEPTS

}



int main(int, char* []) 
{
    using namespace mtl;
    using namespace std;

    typedef mod_n_t<unsigned, 5>    mod_5;
    math::mult<mod_5>               mult_mod_5;

    cout << "equal_results(mod_5(2), mod_5(3), mod_5(4), mod_5(4), mult_mod_5) "
	 << equal_results(mod_5(2u), mod_5(3u), mod_5(4u), mod_5(4u), mult_mod_5) << endl;

    cout << "algebraic_division(mod_5(4), mod_5(2), mult_mod_5) "
	 << algebraic_division(mod_5(4u), mod_5(2u), mult_mod_5) << endl; 

    typedef mod_n_t<unsigned, 28>    mod_28;
    math::mult<mod_28>               mult_mod_28;
    math::add<mod_28>                add_mod_28;
    
    cout << "1/3 " << math::inverse(mult_mod_28, mod_28(3)) 
	 << " check " << math::inverse(mult_mod_28, mod_28(3)) * mod_28(3) << endl;
    cout << "1/5 " << math::inverse(mult_mod_28, mod_28(5)) 
	 << " check " << math::inverse(mult_mod_28, mod_28(5)) * mod_28(5) << endl;
    cout << "1/9 " << math::inverse(mult_mod_28, mod_28(9)) 
	 << " check " << math::inverse(mult_mod_28, mod_28(9)) * mod_28(9) << endl;
    cout << "mod_28 is field: " << (math::is_field(mod_28(0)) ? "true\n" : "false\n");

    cout << "gcd(24, 28): " << gcd(24, 28) << endl;
    cout << "gcd(25, 28): " << gcd(25, 28) << endl;
    

    typedef mod_n_t<unsigned, 127>  mod_127;
    math::mult<mod_127>             mult_mod_127;
    math::add<mod_127>              add_mod_127;

    mod_127   v78(78), v113(113), v90(90), v80(80);
   
    cout << "equal_results(v78, v113,  v90, v80, mult_mod_127) "
         << equal_results(v78, v113,  v90, v80, mult_mod_127) << endl;
    cout << "equal_results(v78, v113,  v90, mod_127(81), mult_mod_127) "
         << equal_results(v78, v113,  v90, mod_127(81), mult_mod_127) << endl;

    cout << "identity_pair(v78, mod_127(-78), mult_mod_127) "
         << identity_pair(v78, mod_127(-78), mult_mod_127) << endl;
    cout << "identity_pair(v78, mod_127(57), mult_mod_127) "
         << identity_pair(v78, mod_127(57), mult_mod_127) << endl;

    cout << "algebraic_division(mod_127(8), mod_127(2), mult_mod_127) = log_2 (8) "
         << algebraic_division(mod_127(8), mod_127(2), mult_mod_127) << endl;
    cout << "algebraic_division(mod_127(35), v78, mult_mod_127) = log_78 (35) "
         << algebraic_division(mod_127(35), v78, mult_mod_127) << endl;
    
    cout << "multiply_and_square(v78, 8, mult_mod_127) = 78^8 "
	 << multiply_and_square(v78, 8, mult_mod_127) << endl;
    cout << "mod_127 is field: " << (math::is_field(mod_127(0)) ? "true\n" : "false\n");

    return 0;
}














#if 0
concept_map AbelianGroup<math::add<mod_n_t<unsigned int, 127u> >, mod_n_t<unsigned int, 127u> > {}
concept_map CommutativeSemiGroup<math::mult<mod_n_t<unsigned int, 127u> >, mod_n_t<unsigned int, 127u> > {}

concept_map GenericField
       < add< mod_n_t<unsigned, 127> >, 
	 mult< mod_n_t<unsigned, 127> >, 
	 mod_n_t<unsigned, 127> 
       > 
{
    // Why do we need the typedefs???
    
    typedef mod_n_t<unsigned, 127> inverse_result_type;
    typedef mod_n_t<unsigned, 127> identity_result_type;
    typedef bool          is_invertible_result_type;
}
#endif 


