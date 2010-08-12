// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef BOOST_MSVC
# pragma warning(disable:4996) // warning C4996: 'std::_Equal' was declared deprecated
#endif 
#include <boost/sequence/algorithm/copy.hpp>
#include <boost/sequence/operations/category.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/test/unit_test.hpp>
#include <algorithm>

namespace sequence = boost::sequence;

template <class T>
void check_integral_constant(T const&)
{
    BOOST_MPL_ASSERT((sequence::detail::is_mpl_integral_constant<T>));
}

template <class Elements, class Begin, class End, class Size>
bool expect_fixed_size_range(
    sequence::range<Elements,Begin,End,Size> const& x
)
{
    typedef sequence::range<Elements,Begin,End,Size> r;

    BOOST_MPL_ASSERT((boost::is_same<typename sequence::intrinsic::tag<r>::type,sequence::range_::tag>));
    
    BOOST_MPL_ASSERT((boost::is_same<typename sequence::intrinsic::tag<r const>::type,sequence::range_::tag>));
    
    BOOST_MPL_ASSERT(
        (boost::is_convertible<
           typename sequence::category<r>::type
         , sequence::fixed_size::category>
        ));
    
    check_integral_constant(sequence::size(x));
    
    return Size::value != 0;
}

int test_main( int, char*[] )
{
    boost::array<char,6> const hello = {{'h','e','l','l','o','\0'}};
    boost::array<char,11> buf = {{'0','1','2','3','4','5','6','7','8','9','\0'}};
    boost::array<char,11> buf2;

    expect_fixed_size_range(
        sequence::algorithm::copy(hello, buf)
    );

    sequence::algorithm::copy(
        sequence::algorithm::copy(hello, buf)
      , buf2
    );

    boost::array<char,11> const result = {{'h','e','l','l','o','\0','6','7','8','9','\0'}};
    BOOST_REQUIRE(
        std::equal(&buf[0],&buf[0]+buf.size(), &result[0] )
    );
    BOOST_REQUIRE(
        std::equal(&buf2[0],&buf2[0]+5,"6789")
    );
    
    return 0;
}

