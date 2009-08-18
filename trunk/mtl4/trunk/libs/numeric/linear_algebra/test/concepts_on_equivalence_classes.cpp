#include <iostream>
#include <cmath>


#ifdef __GXX_CONCEPTS__
#  include <concepts>
#  include <boost/numeric/linear_algebra/new_concepts.hpp>
#else 
#  include <boost/numeric/linear_algebra/pseudo_concept.hpp>
#endif

// Some Vector types
// =================

struct vec
{
    vec (int s) : s(s) {}
    int s;
};

// Just to have another type
template <typename Value>
struct svec
{
    svec (int s) : s(s), p(new Value[s]) {}
    ~svec() { delete[](p); }
    int s;
  private:
    Value* p;
};


template <typename E1, typename E2>
struct vec_add_expr
{
    vec_add_expr(const E1& e1, const E2& e2) : e1(e1), e2(e2) {}
    vec_add_expr(const vec_add_expr& v) : e1(v.e1), e2(v.e2) {}
  private:
    const E1& e1;
    const E2& e2;
};



struct add 
{
    template <typename E1, typename E2>
    vec_add_expr<E1, E2> operator()(const E1& e1, const E2& e2)
    {
	return vec_add_expr<E1, E2>(e1, e2);
    }
};

template <typename Vec>
Vec inline identity(add, const Vec& v) { return v; }


auto concept ConstructibleFromConstRef<typename T, typename U>
{
    T::T(const U&);
}

// Concept for monoid on equivalence classes
// =========================================

concept InMonoidClass<typename Operation, typename Element1, typename Element2>
{
    requires math::Monoid<Operation, Element1>;
    requires math::Monoid<Operation, Element2>;

    requires std::Callable2<Operation, Element1, Element2>;

    typename result_type= std::Callable2<Operation, Element1, Element2>::result_type;

    // I shouldn't need this! 
    // result_type::result_type(const std::Callable2<Operation, Element1, Element2>::result_type&);
    requires ConstructibleFromConstRef<result_type, std::Callable2<Operation, Element1, Element2>::result_type>;

    requires math::Monoid<Operation, result_type>;
}


// Concept and map for equivalence class of real-valued vectors
// ============================================================

concept RealVectorClass<typename Vector> {}


concept_map RealVectorClass<vec> {}

template <typename Value> 
concept_map RealVectorClass<svec<Value> > {}

template <RealVectorClass V1, RealVectorClass V2>
concept_map RealVectorClass<vec_add_expr<V1, V2> > {}


// Concept map for monoid on equivalence classes
// =============================================

namespace math {

#if 0 // would have been too easy
    template <RealVectorClass V> requires std::CopyConstructible<V>
    concept_map Monoid< ::add, V> {}
#endif

    concept_map Monoid< ::add, vec> {}

    template <typename Value> 
    concept_map Monoid< ::add, svec<Value> > 
    {
	typedef vec_add_expr<svec<Value>, svec<Value> > result_type;
	typedef svec<Value>                             identity_result_type;
    }

    template <RealVectorClass V1, RealVectorClass V2>
    concept_map Monoid< ::add, vec_add_expr<V1, V2> > 
    {
	typedef vec_add_expr<vec_add_expr<V1, V2>, vec_add_expr<V1, V2> > result_type;
	typedef vec_add_expr<V1, V2>                                      identity_result_type;
    }
        
}

template <RealVectorClass V1, RealVectorClass V2>
concept_map InMonoidClass<add, V1, V2> {}








template <typename T>
bool is_monoid(const T& x)
{
    return false;
}

template <typename T>
  requires math::Monoid<add, T>
bool is_monoid(const T& x)
{
    return true;
}

template <typename T>
void f(const T& x, const char* name)
{
    std::cout << name << (is_monoid(x) ? " is" : " isn't") << " an additive monoid\n";
}

template <typename Op, typename T, typename U>
  requires InMonoidClass<Op, T, U>
    && std::CopyConstructible<math::Monoid<Op, InMonoidClass<Op, T, U>::result_type>::identity_result_type>
void g(Op op, const T& x, const U& y)
{
    typedef InMonoidClass<Op, T, U>::result_type result_type;

    result_type z(op(x, y));

    math::Monoid<Op, result_type>::identity_result_type id_z(identity(op, z));
}


template <RealVectorClass T> 
void h(const T& x) {}

int main(int, char* [])  
{
    vec           u(3), v(identity(add(), u));
    svec<float>   w(3);

    typedef vec_add_expr<vec, svec<float> > s1_type;
    s1_type                                 s1(u, w);

    typedef vec_add_expr<s1_type, s1_type>  s2_type;
    s2_type                                 s2(s1, s1);

    f(s2, "s2");
    g(add(), s1, s2);
    h(s2);

    return 0;
}
