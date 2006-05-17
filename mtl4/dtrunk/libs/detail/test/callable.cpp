// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/detail/callable.hpp>
#include <boost/detail/lightweight_test.hpp>

struct simple
  : boost::detail::callable<simple, int>
{
    typedef int result_type;

    result_type call() const
    {
        return 1;
    }
    
    template <class A0>
    result_type call(A0& x0) const
    {
        return 2;
    }
    
    template <class A0, class A1>
    result_type call(A0& x0, A1& x1) const
    {
        return 3;
    }
};

int main()
{
    simple f;

    int x = 0;
    int const y = 1;

    f();
    BOOST_TEST(f() == 1);
    BOOST_TEST(f(x) == 1);
    BOOST_TEST(f(y) == 1);
    BOOST_TEST(f(x,y) == 2);
    BOOST_TEST(f(y,x) == 2);
}
