// $COPYRIGHT$

#include <iostream>
#include <boost/test/minimal.hpp>
#include <concepts>


// using namespace std;

template <typename T>
  requires std::Regular<T>
struct complex
{
    template <typename U> requires std::CopyAssignable<T, U>
    complex<T>& operator=(const U& s);
};


int test_main(int argc, char* argv[])
{
    complex<int> cc;

    std::cout << "Hey concepts.\n";

    return 0;
}
