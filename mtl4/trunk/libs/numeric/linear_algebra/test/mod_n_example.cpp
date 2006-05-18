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

    explicit mod_n_t(T const& v) : value(v) {}    

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
};

template<typename T, T N>
inline ostream& operator<< (ostream& stream, const mod_n_t<T, N>& a) 
{
    return stream << a.get(); 
}

template<typename T, T N>
inline bool operator==(const mod_n_t<T, N>& x, const mod_n_t<T, N>& y) 
{
    return x.get() == y.get(); 
} 

template<typename T, T N>
inline bool operator!=(const mod_n_t<T, N>& x, const mod_n_t<T, N>& y) 
{
    return x.get() != y.get(); 
} 

template<typename T, T N>
inline bool operator>=(const mod_n_t<T, N>& x, const mod_n_t<T, N>& y) 
{
    return x.get() >= y.get(); 
} 

template<typename T, T N>
inline bool operator<=(const mod_n_t<T, N>& x, const mod_n_t<T, N>& y) 
{
    return x.get() <= y.get(); 
} 

template<typename T, T N>
inline bool operator>(const mod_n_t<T, N>& x, const mod_n_t<T, N>& y) 
{
    return x.get() > y.get(); 
} 

template<typename T, T N>
inline bool operator<(const mod_n_t<T, N>& x, const mod_n_t<T, N>& y) 
{
    return x.get() < y.get(); 
} 

template<typename T, T N>
inline mod_n_t<T, N> operator+ (const mod_n_t<T, N>& x, const mod_n_t<T, N>& y) 
{
    return mod_n_t<T, N>(x.get() + y.get()); 
} 

template<typename T, T N>
inline mod_n_t<T, N> operator- (const mod_n_t<T, N>& x, const mod_n_t<T, N>& y) 
{
    // add n to avoid negative numbers
    return mod_n_t<T, N>(N + x.get() - y.get()); 
} 

template<typename T, T N>
inline mod_n_t<T, N> operator* (const mod_n_t<T, N>& x, const mod_n_t<T, N>& y) 
{
    return mod_n_t<T, N>(x.get() * y.get()); 
} 

template<typename T, T N>
inline mod_n_t<T, N> operator/ (const mod_n_t<T, N>& x, const mod_n_t<T, N>& y) 
{
    if (y.get() == 0) throw "Division by 0";

    T u= y.get(), v= N, x1= 1, x2= 0, q, r, x0;
  
    while (u != 1) {
	q= v/u; 
	r= v%u; 
	x0= x2 - q*x1; 
	v= u; 
	u= r; 
	x2= x1; 
	x1= x0;
    }
    return mod_n_t<T, N>(x.get() * x1); 
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

    typedef mod_n_t<unsigned, 127>  mod_127;
    math::mult<mod_127>             mult_mod_127;

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
    
    return 0;
}
