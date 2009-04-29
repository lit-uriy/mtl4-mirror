// $COPYRIGHT$

#include <iostream>
#include <boost/test/minimal.hpp>
#include <concepts>

#include <boost/numeric/complex/complex.hpp>

using namespace std;


int h(int x, int y) { return x + y; }

// Regular procedure essentially means that the result does only depend on the arguments
// not on other values -- p 54

// Op returns a constant 
concept RegularProcedure0<typename Op> 
  : Callable0<Op>
{}

// Op doesn't depend on values other than the arguments
concept RegularProcedure1<typename Op, typename X> 
  : Callable1<Op, X>
{}

// Op doesn't depend on values other than the arguments
concept RegularProcedure2<typename Op, typename X, typename Y> 
  : Callable2<Op, X, Y>
{}

#if 0 // Variadic version
// Op doesn't depend on values other than the argument(s)
concept RegularProcedure<typename Op, typename... Args> 
  : Callable<Op, Args...>
{}
#endif

// Define it only for one argument and variadic -- p 56
auto concept FunctionalProcedure1<typename Op, typename X>
  : RegularProcedure1<Op, X>
{
    requires Regular<X>;
}

#if 0 // Variadic version
auto concept FunctionalProcedure<typename Op, typename... Args>
  : RegularProcedure<Op, Args...>
{
    requires Regular<Args...>;
}
#endif

#if 0 
// Variadic equality of types
concept SameTypes<typename... X> {}

template <typename X> concept_map SameTypes<X> {}
template <typename X, typename... Tail> 
    requires SameTypes<X, Tail...> 
concept_map SameTypes<X, X, Tail...> {}

// Maybe it should be called HomogeneousFunctionCall
concept HomogeneousFunction<typename Op, typename... Args> // p 71
  : RegularProcedure<Op, Args...>
{
    requires SameTypes<Args...>;
}

#endif

// Can be defined without the concept Arity -- p 73
template <typename X, typename Op>
requires Callable2<Op, X, X> && CopyConstructible<Callable2<Op, X, X>::result_type>
Callable2<Op, X, X>::result_type square(const X& x, Op op)
{
    return op(x, x);
}

auto concept RegularUnaryFunction<typename Op, typename X> // p 76
  : FunctionalProcedure1<Op, X>
{
    requires Regular<FunctionalProcedure1<Op, X>::result_type>;
 
    // Normally, concepts with axioms must not be auto because the semantics can not be deduced from the syntax
    // However, this is a diffent case because the semantics is implied by the required and derived concepts
    axiom Uniqueness(Op f, Op fp, X x, X xp)
    {
	if (f == fp && x == xp)
	    f(x) == fp(xp);
    }
}


int test_main(int argc, char* argv[])
{
    std::cout << "square(3, h) is " << square(3, h) << '\n';

    return 0;
}
