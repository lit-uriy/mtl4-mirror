// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_RANGE_DWA200559_HPP
# define BOOST_SEQUENCE_RANGE_DWA200559_HPP

# include <boost/sequence/range_fwd.hpp>
# include <boost/compressed_pair.hpp>
# include <boost/sequence/intrinsic/tag_fwd.hpp>
# include <boost/sequence/homogeneous.hpp>
# include <boost/sequence/category_fwd.hpp>
# include <boost/sequence/fixed_size/category.hpp>
# include <boost/sequence/fixed_size/intrinsics.hpp>

namespace boost { namespace sequence { 

#ifdef BOOST_MSVC
# pragma warning(push)
# pragma warning(disable:4512)
#endif 

namespace range_
{
  struct accessor;
  
  template <class Elements, class Begin, class End>
  class range
    : private compressed_pair<
          Begin
        , compressed_pair<
              End
            , Elements
          >
      >
  {
      friend struct accessor;
      
      typedef compressed_pair<
          Begin
        , compressed_pair<
              End
            , Elements
          >
      > base;

   public:
      typedef Elements elements;
      typedef Begin begin;
      typedef End end;
      
      range(Elements const& m, Begin const& b, End const& e)
        : base(b, compressed_pair<End,Elements>(e,m))
      {}

  };

  struct accessor
  {
      template <class R>
      static typename R::base data(R const& r)
      { return r; }
  };

  // Identifies ranges for tag dispatch purposes
  struct tag {};
}

#ifdef BOOST_MSVC
# pragma warning(pop)
#endif 

using range_::range;

namespace intrinsic
{
  template <class Elements, class Begin, class End>
  struct tag<range<Elements,Begin,End> >
  {
      typedef range_::tag type;
  };

  template <class Range>
  struct operations<Range, range_::tag>
  {
      struct begin
      {
          typedef typename Range::begin type;
          type operator()(Range& r)
          {
              return range_::accessor::data(r).first();
          }
      };

      struct end
      {
          typedef typename Range::end type;
          type operator()(Range& r)
          {
              return range_::accessor::data(r).second().first();
          }
      };

      struct elements
      {
          typedef typename Range::elements type;
          type operator()(Range& r)
          {
              return range_::accessor::data(r).second().second();
          }
      };
  };
}

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_RANGE_DWA200559_HPP
