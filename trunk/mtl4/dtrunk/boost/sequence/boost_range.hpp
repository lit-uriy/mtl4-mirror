// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_BOOST_RANGE_DWA2006513_HPP
# define BOOST_SEQUENCE_BOOST_RANGE_DWA2006513_HPP

# include <boost/sequence/tag.hpp>
# include <boost/sequence/intrinsics.hpp>
# include <boost/property_map/dereference.hpp>
# include <boost/range/result_iterator.hpp>
# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>
# include <boost/iterator/counting_iterator.hpp>

namespace boost { namespace sequence { 

struct boost_range_tag {};

namespace impl
{
  template <class S>
  struct tag
  {
      // By default assume all sequences are models of the boost Range
      // concept.
      typedef boost_range_tag type;
  };

  template <class S>
  struct intrinsics<S, boost_range_tag>
  {
      typedef counting_iterator<
          typename range_result_iterator<S>::type
      > cursor;
      
      struct begin
      {
          typedef cursor result_type;
          
          result_type operator()(S& s) const
          {
              return result_type(boost::begin(s));
          }
      };

      struct end
      {
          typedef cursor result_type;
          
          result_type operator()(S& s) const
          {
              return result_type(boost::end(s));
          }
      };

      struct elements
      {
          typedef property_map::dereference result_type;
      
          result_type operator()(S& s) const
          {
              return result_type();
          }
      };
  };
}

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_BOOST_RANGE_DWA2006513_HPP
