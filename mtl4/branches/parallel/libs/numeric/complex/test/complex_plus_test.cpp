// $COPYRIGHT$

#include <iostream>
#include <boost/test/minimal.hpp>
#include <concepts>

#include <boost/numeric/complex/complex.hpp>

using namespace std;

template <typename X, typename Y> 
void test2(X& x, const char* xname, Y& y, const char* yname)
{
    cout << "Adding " << xname << " and " << yname << "\n";
    x= 2.0; y= 3.0;
    cout << "Result is: " << x + y << " (typeid == " << typeid(x+y).name() << ")\n";

    if (x + y != 5.) throw "Result should be 5!";
}

template <typename X> 
void test(X& x, const char* xname)
{
    float                   f;
    newstd::complex<float>  c;
    double                  d;
    newstd::complex<double> z;

    test2(x, xname, f, "float");
    test2(x, xname, c, "complex<float>");
    test2(x, xname, d, "double");
    test2(x, xname, z, "complex<double>");
}


int test_main(int argc, char* argv[])
{
    float                   f;
    newstd::complex<float>  c;
    double                  d;
    newstd::complex<double> z;

    test(f, "float");
    test(c, "complex<float>");
    test(d, "double");
    test(z, "complex<double>");

    return 0;
}
