// $COPYRIGHT$

#include <iostream>
#include <boost/test/minimal.hpp>
#include <concepts>


// using namespace std;

template <typename T>
    requires std::Regular<T>
struct complex
{
    // leads to infinit loop
    template <typename U> complex (const complex<U>& z) {}
};


int test_main(int argc, char* argv[])
{
    complex<int> cc;
    return 0;
}
