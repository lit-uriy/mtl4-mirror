// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef IS_TUPLE_DWA200555_HPP
# define IS_TUPLE_DWA200555_HPP

# include <boost/sequence/category_fwd.hpp>
# include <boost/sequence/algorithm/tuple/category.hpp>
# include <boost/type_traits/is_convertible.hpp>
# include <cstddef>

namespace boost { namespace sequence { namespace tuple {

template <class T>
struct is_tuple
  : is_convertible<
        typename category<T>::type
      , algorithm::tuple::category
    >
{};


}}} // namespace boost::sequence::tuple

#endif // IS_TUPLE_DWA200555_HPP
