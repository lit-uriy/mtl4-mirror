// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_WRITER_DWA200655_HPP
# define BOOST_SEQUENCE_WRITER_DWA200655_HPP

# include <boost/detail/function2.hpp>
# include <boost/detail/pod_singleton.hpp>
# include <boost/sequence/tag.hpp>
# include <boost/mpl/placeholders.hpp>

namespace boost { namespace sequence { 

namespace impl
{
  struct indirect_write
  {
      typedef void result_type;
      
      template <class Key, class Value>
      result_type operator()(Key const& key, Value const& v) const
      {
          *p = v;
      }
      
      template <class Key, class Value>
      result_type operator()(Key const& key, Value& v) const
      {
          *p = v;
      }
  };
  
  template <class S, class = typename tag<S>::type>
  struct writer
  {
      // The default cursor is a (STL or fusion) counting_iterator; we
      // write the element at its position by dereferencing it to get
      // a key (which is an iterator), then dereferencing that and
      // assigning the new element value.  The function object
      // returned by writer(s) supplies the 2nd dereference and the
      // assignment.
      typedef indirect_write result_type;
      
      result_type operator()(S& s) const
      {
          return result_type();
      }
  };
}

namespace op
{
  using mpl::_;
  struct writer : boost::detail::function1<impl::writer<_,impl::tag<_> > > {};
}

namespace
{
  op::writer const& writer = boost::detail::pod_singleton<op::writer>::instance;
}

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_WRITER_DWA200655_HPP
