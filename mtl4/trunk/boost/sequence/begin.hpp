// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_BEGIN_DWA200655_HPP
# define BOOST_SEQUENCE_BEGIN_DWA200655_HPP

# include <boost/range/iterator.hpp>
# include <boost/detail/function1.hpp>

namespace boost { namespace sequence { 

namespace impl
{
  template <class S>
  struct begin
  {
      typedef counting_iterator<
          typename range_iterator<S>::type
      > result_type;
      
      result_type operator()(S& s) const
      {
          return boost::begin(s);
      }
  };
}

namespace op
{
  struct begin : boost::detail::function1<impl::begin> {};
}

namespace
{
  op::begin const& begin = boost::detail::pod_singleton<op::begin>::instance;
}

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_BEGIN_DWA200655_HPP
