// $COPYRIGHT$

#include <iostream>
#include <boost/test/minimal.hpp>
#include <concepts>

#include <boost/numeric/complex/complex.hpp>

using namespace std;


concept Arity<typename F>
{
    int arity(F);
}

template <Callable0 F> concept_map Arity<F>
{
    int arity(const F& f) { return 0; }
}

#if 0
// Doesn't work, X must be a concept argument not an unbound parameter
template <typename F, typename X> requires Callable1<F, X> concept_map Arity<F>
{
    int arity(const F& f) { return 1; }
}
#endif


double f0() { return 7.0; }
double f1(char c) { return 7.0; }


struct g
{
    int operator()() { return 4; }
    int operator()(char c) { return c; }
};


template <typename F> 
  requires Arity<F>
void test(const F& f, int a, const char* name)
{
    std::cout << "Function " << name << " has arity " << arity(f) << "\n";
    if (arity(f) != a) throw "Wrong arity.";
}


int test_main(int argc, char* argv[])
{
    test(f0, 0, "f0");
    //test(f1, 1, "f1");
    test(g(), 0, "g");

    return 0;
}
