// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/sequence/size.hpp>
#include <boost/sequence/minus.hpp>
#include <boost/sequence/detail/is_mpl_integral_constant.hpp>
#include <boost/mpl/assert.hpp>

namespace sequence = boost::sequence;

typedef boost::array<char,11> a;
typedef sequence::intrinsic::size< a >::type s1;
typedef sequence::intrinsic::size< a const >::type s2;

BOOST_MPL_ASSERT((sequence::detail::is_mpl_integral_constant<s1>));
BOOST_MPL_ASSERT((sequence::detail::is_mpl_integral_constant<s2>));

typedef sequence::intrinsic::minus<s1,s2>::type d;

BOOST_MPL_ASSERT((sequence::detail::is_mpl_integral_constant<d>));
