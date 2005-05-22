// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#include <boost/sequence/end_cursor.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/type_traits/is_same.hpp>

using namespace boost;

BOOST_MPL_ASSERT((is_same<sequence::end_cursor<char const[3]>::type, sequence::fixed_size::cursor<3> >));
BOOST_MPL_ASSERT((is_same<sequence::end_cursor<char[3]>::type, sequence::fixed_size::cursor<3> >));
