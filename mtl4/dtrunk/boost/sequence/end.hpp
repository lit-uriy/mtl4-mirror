// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_END_DWA200655_HPP
# define BOOST_SEQUENCE_END_DWA200655_HPP

# include <boost/detail/function1.hpp>
# include <boost/detail/pod_singleton.hpp>
# include <boost/sequence/boost_range.hpp>

namespace boost { namespace sequence { 

namespace impl
{
  template <class S>
  struct end
    : intrinsics<S>::end
  {};
}

namespace op
{
  struct end : boost::detail::function1<impl::end> {};
}

namespace
{
  op::end const& end = boost::detail::pod_singleton<op::end>::instance;
}

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_END_DWA200655_HPP
