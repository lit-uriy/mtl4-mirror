// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_CATEGORY_DWA200559_HPP
# define BOOST_SEQUENCE_CATEGORY_DWA200559_HPP

# include <boost/sequence/category_fwd.hpp>
# include <boost/sequence/detail/o1_size_cursors.hpp>
# include <boost/sequence/detail/is_mpl_integral_constant.hpp>
# include <boost/sequence/fixed_size/category.hpp>
# include <boost/sequence/intrinsic/operations.hpp>

# include <boost/mpl/eval_if.hpp>
# include <boost/mpl/if.hpp>

namespace boost { namespace sequence {

namespace detail
{
  template <class S>
  struct has_constant_size
    : is_mpl_integral_constant<
          typename intrinsic::size<S>::type
      >
  {};
}

template <class Sequence>
struct category_impl
  : mpl::eval_if<
        // If the cursors yield size in O(1) by themselves
        detail::o1_size_cursors<
            typename intrinsic::begin<Sequence>::type
          , typename intrinsic::end<Sequence>::type
        >
      , mpl::if_<
            // We check to see if size yields an integral constant
            detail::has_constant_size<Sequence>
          , fixed_size::category
          , o1_size_tag
        >
      , mpl::identity<sequence_tag>
    >
{};

}}


#endif // BOOST_SEQUENCE_CATEGORY_DWA200559_HPP
