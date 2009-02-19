// $COPYRIGHT$

#include <iostream>
#include <boost/test/minimal.hpp>
#include <concepts>

#include <boost/numeric/complex/complex.hpp>

using namespace std;

template <typename X, typename Y> 
void test2(X& x, const char* xname, Y& y, const char* yname)
{
    cout << xname << " / " << yname << "\n";
    x= 6.0; y= 3.0;
    cout << "Result is: " << x / y << " (typeid == " << typeid(x/y).name() << ")\n";

    typedef typename std::HasMinus<X, Y>::result_type R; // Need this type because float cannot be compared with long double
    if (newstd::norm(x / y - 2.) > 0.001) throw "Result should be 2!";
}

template <typename X> 
void test(X& x, const char* xname)
{
    float                   f;
    newstd::complex<float>  c;
    double                  d;
    newstd::complex<double> z;
    long double             ld;
    newstd::complex<long double> lz;

    test2(x, xname, f, "float");
    test2(x, xname, c, "complex<float>");
    test2(x, xname, d, "double");
    test2(x, xname, z, "complex<double>");
    test2(x, xname, ld, "long double");
    test2(x, xname, lz, "complex<long double>");
}


int test_main(int argc, char* argv[])
{
    float                   f;
    newstd::complex<float>  c;
    double                  d;
    newstd::complex<double> z;
    long double             ld;
    newstd::complex<long double> lz;

    // test(f, "float");
    test(c, "complex<float>");
    // test(d, "double");
    test(z, "complex<double>");
    // test(ld, "long double");
    test(lz, "complex<long double>");

    return 0;
}
