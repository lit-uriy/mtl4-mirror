// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/mpl/assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/sequence/operations/category.hpp>
#include <boost/sequence/class/range/range.hpp>

namespace seq =  boost::sequence;

BOOST_MPL_ASSERT((
  boost::is_same<
      seq::category<seq::range<int,int,char> >::type
    , seq::algorithm::fixed_size::category
  >
                 ));
