// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_TRAITS_IS_HOMOGENEOUS_DWA2006926_HPP
# define BOOST_SEQUENCE_TRAITS_IS_HOMOGENEOUS_DWA2006926_HPP

# include <boost/fusion/support/is_sequence.hpp>
# include <boost/mpl/not.hpp>

namespace boost { namespace sequence { namespace traits { 

template <class S>
struct is_homogeneous
  : mpl::not_<fusion::traits::is_sequence<S> >
{};
      
}}} // namespace boost::sequence::traits

#endif // BOOST_SEQUENCE_TRAITS_IS_HOMOGENEOUS_DWA2006926_HPP
