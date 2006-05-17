// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/property_map/dereference.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/detail/lightweight_test.hpp>
#include <algorithm>

int x[] = {3,1,4,1,5,9,2,7};
std::size_t const len = sizeof(x)/sizeof(*x);
int a0[len] = {};
int a1[len] = {};
int a2[len] = {};
int a3[len] = {};

template <class F1, class I1, class F2, class I2>
void copy1(F1 const& srcf, I1 srcb, I1 srce, F2 const& dstf, I2 dstb)
{
    while (srcb != srce)
    {
        dstf( *dstb, srcf( *srcb ) );
        ++srcb;
        ++dstb;
    }
}

template <class F1, class I1, class F2, class I2>
void copy2(F1 srcf, I1 srcb, I1 srce, F2 dstf, I2 dstb)
{
    while (srcb != srce)
    {
        dstf( *dstb, srcf( *srcb ) );
        ++srcb;
        ++dstb;
    }
}

int main()
{
    typedef boost::counting_iterator<int*> i;
    typedef boost::counting_iterator<int const*> ci;
    typedef boost::property_map::dereference d;
    
    copy1( d(), i(x), i(x+len), d(), i(a0) );
    BOOST_TEST(std::equal(x,x+len,a0));
    
    copy1( d(), ci(x), ci(x+len), d(), i(a1) );
    BOOST_TEST(std::equal(x,x+len,a1));

    copy2( d(), i(x), i(x+len), d(), i(a2) );
    BOOST_TEST(std::equal(x,x+len,a2));
    
    copy2( d(), ci(x), ci(x+len), d(), i(a3) );
    BOOST_TEST(std::equal(x,x+len,a3));
}
