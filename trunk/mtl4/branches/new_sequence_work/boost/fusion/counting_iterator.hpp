// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_FUSION_COUNTING_ITERATOR_DWA2006919_HPP
# define BOOST_FUSION_COUNTING_ITERATOR_DWA2006919_HPP

# include <boost/fusion/support/iterator_base.hpp>
# include <boost/fusion/support/category_of.hpp>
# include <boost/detail/transfer_cv.hpp>
# include <boost/type_traits/add_reference.hpp>
# include <boost/fusion/iterator/next.hpp>
# include <boost/fusion/iterator/prior.hpp>
# include <boost/fusion/iterator/deref.hpp>
# include <boost/fusion/iterator/distance.hpp>
# include <boost/fusion/iterator/advance.hpp>

namespace boost { namespace fusion { 

template <class Iterator>
struct counting_iterator
  : iterator_base<counting_iterator<Iterator> >
{
    typedef typename category_of<Iterator>::type category;
    typedef Iterator base_type;
    
    counting_iterator(Iterator base)
        : base(base) {}

    Iterator const base;
};

struct counting_iterator_tag {};

namespace traits
{
  template<class Iterator>
  struct tag_of<counting_iterator<Iterator> >
  {
      typedef counting_iterator_tag type;
  };
}

namespace extension
{
  template<>
  struct value_of_impl<counting_iterator_tag>
  {
      template<class Iterator>
      struct apply
      {
          typedef typename Iterator::base_type type;
      };
  };

  template<>
  struct deref_impl<counting_iterator_tag>
  {
      template<typename Iterator>
      struct apply
        : add_reference<
              typename detail::transfer_cv<Iterator,typename Iterator::base_type>::type
          >
      {
          static typename apply::type
          call(Iterator const& it)
          {
              return it.base;
          }
      };
  };

  template<>
  struct next_impl<counting_iterator_tag>
  {
      template<typename Iterator>
      struct apply
        : counting_iterator<typename fusion::result_of::next<typename Iterator::base_type>::type>
      {
          static typename apply::type
          call(Iterator const& it)
          {
              return apply::type(fusion::next(it.base));
          }
      };
  };

  template<>
  struct equal_to_impl<counting_iterator_tag>
  {
      template<typename Iterator0, typename Iterator1>
      struct apply
        : fusion::result_of::equal_to<typename Iterator0::base_type,typename Iterator1::base_type>
      {
          static typename apply::type
          call(Iterator0 const& i0, Iterator1 const& i1)
          {
              return fusion::equal_to(i0, i1);
          }
      };
  };

  template<>
  struct advance_impl<counting_iterator_tag>
  {
      template<typename Iterator, typename N>
      struct apply
        : counting_iterator<typename fusion::result_of::advance<typename Iterator::base_type>::type>
      {
          static typename apply::type
          call(Iterator const& it)
          {
              return apply::type(fusion::advance<N>(it.base));
          }
      };
  };

  template<>
  struct distance_impl<counting_iterator_tag>
  {
      template<typename First, typename Last>
      struct apply
        : fusion::result_of::distance<typename First::base_type,typename Last::base_type>
      {
          static typename apply::type
          call(First const& first, Last const& last)
          {
              return fusion::distance(first.base, last.base);
          }
      };
  };
}

}} // namespace boost::fusion

#endif // BOOST_FUSION_COUNTING_ITERATOR_DWA2006919_HPP
