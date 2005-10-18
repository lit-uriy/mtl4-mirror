// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef IS_FIXED_SIZE_DWA200555_HPP
# define IS_FIXED_SIZE_DWA200555_HPP

# include <boost/sequence/category_fwd.hpp>
# include <boost/sequence/algorithm/fixed_size/category.hpp>
# include <boost/type_traits/is_convertible.hpp>
# include <cstddef>

namespace boost { namespace sequence { namespace fixed_size {

template <class T>
struct is_fixed_size
  : is_convertible<
        typename category<T>::type
      , algorithm::fixed_size::category
    >
{};


}}} // namespace boost::sequence::fixed_size

#endif // IS_FIXED_SIZE_DWA200555_HPP
