#include <iostream>

#include <boost/numeric/linear_algebra/identity.hpp>
#include <boost/numeric/linear_algebra/is_invertible.hpp>
#include <boost/numeric/linear_algebra/inverse.hpp>
#include <boost/numeric/linear_algebra/operators.hpp>
#include <boost/numeric/linear_algebra/concepts.hpp>

#include "algebraic_functions.hpp"

using std::ostream;


template<typename T, T N>
// where Integral<T>
class mod_n_t 
{
    // or BOOST_STATIC_ASSERT((IS_INTEGRAL))

    T                 value;
 public:
    typedef T         value_type;
    typedef mod_n_t   self;

    static const T modulo= N;

    explicit mod_n_t(T const& v) : value( v % modulo ) {
	// std::cout << "v " << v << ", modulo " << modulo  << ", value " << value  << "\n";
    }    

    // modulo of negative numbers can be bizarre
    // better use constructor for T
    explicit mod_n_t(int v) 
    {
	// std::cout << "v " << v << ", modulo " << modulo << ", v%modulo " << v%modulo << "\n";
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

template<typename T, T N>
inline bool operator==(const mod_n_t<T, N>& x, const mod_n_t<T, N>& y) 
{
    check(x); check(y);
    return x.get() == y.get(); 
} 

template<typename T, T N>
inline bool operator!=(const mod_n_t<T, N>& x, const mod_n_t<T, N>& y) 
{
    check(x); check(y);
    return x.get() != y.get(); 
} 

template<typename T, T N>
inline bool operator>=(const mod_n_t<T, N>& x, const mod_n_t<T, N>& y) 
{
    check(x); check(y);
    return x.get() >= y.get(); 
} 

template<typename T, T N>
inline bool operator<=(const mod_n_t<T, N>& x, const mod_n_t<T, N>& y) 
{
    check(x); check(y);
    return x.get() <= y.get(); 
} 

template<typename T, T N>
inline bool operator>(const mod_n_t<T, N>& x, const mod_n_t<T, N>& y) 
{
    check(x); check(y);
    return x.get() > y.get(); 
} 

template<typename T, T N>
inline bool operator<(const mod_n_t<T, N>& x, const mod_n_t<T, N>& y) 
{
    check(x); check(y);
    return x.get() < y.get(); 
} 

template<typename T, T N>
inline mod_n_t<T, N> operator+ (const mod_n_t<T, N>& x, const mod_n_t<T, N>& y) 
{
    check(x); check(y);
    return mod_n_t<T, N>(x.get() + y.get()); 
} 

template<typename T, T N>
inline mod_n_t<T, N> operator- (const mod_n_t<T, N>& x, const mod_n_t<T, N>& y) 
{
    check(x); check(y);
    // add n to avoid negative numbers
    return mod_n_t<T, N>(N + x.get() - y.get()); 
} 

template<typename T, T N>
inline mod_n_t<T, N> operator* (const mod_n_t<T, N>& x, const mod_n_t<T, N>& y) 
{
    check(x); check(y);

    return mod_n_t<T, N>(x.get() * y.get()); 
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
#if 0
template<typename T, T N>
inline mod_n_t<T, N> operator/ (const mod_n_t<T, N>& x, const mod_n_t<T, N>& y) 
{
    check(x); check(y);
    if (y.get() == 0) throw "Division by 0";

    int u= N, v= y.get(), u1= 1, u2= 0, v1= 0, v2= 1, q, r, r1, r2;

    while (u % v != 0) {
	q= u / v;

	r= u % v; /* r1= u1 - q * v1; */ r2= u2 - q * v2;
	u= v; /* u1= v1; */ u2= v2;
	v= r; /* v1= r1; */ v2= r2;
    }

    return x * mod_n_t<T, N>(v2); 
} 
#endif


template<typename T, T N>
inline mod_n_t<T, N> operator/ (const mod_n_t<T, N>& x, const mod_n_t<T, N>& y) 
{
    if (y.get() == 0) throw "Division by 0";

    int u= y.get(), v= N, x1= 1, x2= 0, q, r, x0;
  
    while (v % u != 0) {
	q= v/u; 
	r= v%u; 
	x0= x2 - q*x1; 
	v= u; 
	u= r; 
	x2= x1; 
	x1= x0;
    }
    return x * mod_n_t<T, N>(x1); 
} 



namespace math {

    template<typename T, T N>
    struct identity< mult< mod_n_t<T, N> >, mod_n_t<T, N> >
    {
	mod_n_t<T, N> operator() (mod_n_t<T, N> const& v) const
	{
	    return mod_n_t<T, N>(1);
	}
    };


    // Reverse definition, a little more efficient if / uses inverse
    template<typename T, T N>
    struct inverse< mult< mod_n_t<T, N> >, mod_n_t<T, N> >
    {
	mod_n_t<T, N> operator() (mod_n_t<T, N> const& v) const
	{
	    return mod_n_t<T, N>(1) / v;
	}
    };
    

    template<typename T, T N>
    struct is_invertible< mult< mod_n_t<T, N> >, mod_n_t<T, N> >
    {
	bool operator() (mod_n_t<T, N> const& v) const
	{
	    T value = v.get();
	    return value != 0 && N%value != 0;
	}
    };
    


# ifdef LA_WITH_CONCEPTS

#if 0 // more operators needed for this
    // All modulo sets are commutative rings with identity
    // but only if N is prime it is also a field
    template <typename T, T N>
    concept_map CommutativeRingWithIdentity< mod_n_t<T, N> > {}

    concept_map Field< mod_n_t<unsigned, 127> > {}
#endif    

    // Can we express Prime<N> as a concept?

    template <typename T, T N>
    concept_map PartiallyInvertibleMonoid< mult< mod_n_t<T, N> >, mod_n_t<T, N> > {}

    // Shouldn't be needed
    template <typename T, T N>
    concept_map Closed2EqualityComparable< mult< mod_n_t<T, N> >, mod_n_t<T, N> > {}

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

    for (unsigned i= 0; i < 5; i++)
      for (unsigned j= 0; j < 5; j++) {
	mod_5 mi(i), mj(j);
	cout << mi << " * " << mj << " = " << mi*mj;
	if (j != 0)
	  cout << ", div = " << mi/mj;
	cout << ", add = " << mi+mj << ", minus = " << mi-mj << "\n";
      }


    typedef mod_n_t<unsigned, 127>  mod_127;
    math::mult<mod_127>             mult_mod_127;


    for (unsigned i= 0; i < 126; i++)
      for (unsigned j= 1; j < 126; j++) {
	mod_127 mi(i), mj(j);
	assert((mi / mj) * mj == mi);
      }

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

    return 0;
}
