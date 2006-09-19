// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_READER_DWA200655_HPP
# define BOOST_SEQUENCE_READER_DWA200655_HPP

# include <boost/detail/function1.hpp>
# include <boost/detail/pod_singleton.hpp>
# include <boost/sequence/tag.hpp>
# include <boost/mpl/placeholders.hpp>
# include <boost/iterator/iterator_traits.hpp>
# include <boost/cursor/deref.hpp>

namespace boost { namespace sequence { 

namespace impl
{
  template <class S, class = typename tag<S>::type>
  struct reader
  {
      // The default cursor is a counting_iterator; we read the
      // element at its position by dereferencing it to get a key, and
      // then dereferencing that.  The function object returned by
      // reader(s) supplies the 2nd dereference.
      typedef cursor::op::deref result_type;
      
      result_type operator()(S& s) const
      {
          return result_type();
      }
  };
}

namespace op
{
  using mpl::_;
  struct reader : boost::detail::function1<impl::reader<_,impl::tag<_> > > {};
}

namespace
{
  op::reader const& reader = boost::detail::pod_singleton<op::reader>::instance;
}

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_READER_DWA200655_HPP
