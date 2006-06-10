#include <concepts>
#include <boost/operators.hpp>


auto concept BinaryIsoFunction<typename Operation, typename Element>
//  : std::Callable2<Operation, Element, Element>
{
   where std::Callable2<Operation, Element, Element>;
    where std::Convertible<std::Callable2<Operation, Element, Element>::result_type, Element>;

    typename result_type = std::Callable2<Operation, Element, Element>::result_type;
};

// auto 
concept Magma<typename Operation, typename Element>
    : BinaryIsoFunction<Operation, Element>
{
    where std::Assignable<Element>;
    where std::Assignable<Element, BinaryIsoFunction<Operation, Element>::result_type>;
};

template <typename Element>
struct add
{
    Element operator() (const Element& x, const Element& y)
    {
	return x + y;
    }
};


// auto 
concept AdditiveMagma<typename Element>
// : Magma< add<Element>, Element >
{
  where Magma< add<Element>, Element >;

    typename assign_result_type;  
    assign_result_type operator+=(Element& x, Element y);

    // Operator + is by default defined with +=
    typename result_type;  
    result_type operator+(Element& x, Element y);
#if 0
    {
	Element tmp(x);
	return tmp += y;                      defaults NYS
    }
#endif 
    
    // Type consistency with Magma
    where std::SameType< result_type, 
	                 Magma< add<Element>, Element >::result_type>;

    axiom Consistency(add<Element> op, Element x, Element y)
    {
	op(x, y) == x + y;                    
	// Might change later
        x + y == x += y;
	// Element tmp = x; tmp+= y; tmp == x + y; not proposal-compliant
    }   
}


concept_map Magma< add<int>, int> {}
concept_map AdditiveMagma<int> {}


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

    // Goes wrong with unsigned b/c some values will be negative
    int u= N, v= y.get(), /* u1= 1, */  u2= 0, /* v1= 0, */  v2= 1, q, r, /* r1, */  r2;

    while (u % v != 0) {
	q= u / v;

	r= u % v; /* r1= u1 - q * v1; */ r2= u2 - q * v2;
	u= v; /* u1= v1; */ u2= v2;
	v= r; /* v1= r1; */ v2= r2;
    }

    return *this *= mod_n_t<T, N>(v2); 
}


    template <typename T, T N>
    concept_map Magma< add< mod_n_t<T, N> >, mod_n_t<T, N> > {}




int main() {
  return 0;
}
