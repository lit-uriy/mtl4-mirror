// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/sequence/core/detail/has_value.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/assert.hpp>

using namespace boost;

struct dummy {};
BOOST_MPL_ASSERT_NOT((sequence::detail::has_value<int>));
BOOST_MPL_ASSERT_NOT((sequence::detail::has_value<dummy>));
BOOST_MPL_ASSERT((sequence::detail::has_value<mpl::int_<5> >));
BOOST_MPL_ASSERT((sequence::detail::has_value<mpl::int_<-5> >));
