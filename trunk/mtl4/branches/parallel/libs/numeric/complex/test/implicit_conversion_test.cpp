// $COPYRIGHT$

#include <iostream>
#include <boost/test/minimal.hpp>
#include <limits>


using namespace std;

template <typename X, typename Y> 
void test2(X& x, const char* xname, Y& y, const char* yname)
{
    cout << "Adding/Multiplying " << xname << "(" << typeid(x).name() << ") and " << yname << "(" << typeid(y).name() << ")"
	 << " results in  " << typeid(x+y).name() << "/" << typeid(x*y).name() << "\n";
}

template <typename X> 
void test(X& x, const char* xname)
{
    char             c(2);
    unsigned char    uc(2);
    short            s(2);
    unsigned short   us(2);
    int              i(2);
    unsigned int     ui(2);
    long             l(2);
    unsigned long    ul(2);
    float            f(2.);
    double           d(2.);
    long double      ld(2.);

    cout << "\nMax of " << xname << " is " << std::numeric_limits<X>::max() << ", max-1 is " << std::numeric_limits<X>::max()-X(1) 
	 << "(" << typeid(std::numeric_limits<X>::max()-X(1)).name() << ")"<< "\n"
	 << "Min of " << xname << " is " << std::numeric_limits<X>::min() << ", min+1 is " << std::numeric_limits<X>::min()+X(1) << "\n";

    test2(x, xname, c, "char");
    test2(x, xname, uc, "unsigned char");
    test2(x, xname, s, "short");
    test2(x, xname, us, "unsigned short");
    test2(x, xname, s, "int");
    test2(x, xname, us, "unsigned int");
    test2(x, xname, s, "long");
    test2(x, xname, us, "unsigned long");
    test2(x, xname, f, "float");
    test2(x, xname, d, "double");
    test2(x, xname, ld, "long double");
}


int test_main(int argc, char* argv[])
{
    char             c(2);
    unsigned char    uc(2);
    short            s(2);
    unsigned short   us(2);
    int              i(2);
    unsigned int     ui(2);
    long             l(2);
    unsigned long    ul(2);
    float            f(2.);
    double           d(2.);
    long double      ld(2.);

    test(c, "char");
    test(uc, "unsigned char");
    test(s, "short");
    test(us, "unsigned short");
    test(s, "int");
    test(us, "unsigned int");
    test(s, "long");
    test(us, "unsigned long");
    test(f, "float");
    test(d, "double");
    test(ld, "long double");

    return 0;
}
