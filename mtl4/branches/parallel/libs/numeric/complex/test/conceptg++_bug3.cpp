// $COPYRIGHT$

#include <iostream>
#include <boost/test/minimal.hpp>
#include <concepts>

concept Con<typename T> {}


template <typename T>
struct foo
{
    template <typename U> 
    foo& operator=(const U& src) {}

#if 0 // Concept-restrictions seems to be ignored for member functions
    template <typename U> requires Con<U>
    foo& operator=(const U& src) {}
#endif
};

template <typename U>
U f(const U& src) { return src;}

template <typename U> requires Con<U> && std::CopyConstructible<U>
U f(const U& src) { return src;}

int test_main(int argc, char* argv[])
{
    foo<int> cc;

    std::cout << "Hey concepts.\n";

    return 0;
}
