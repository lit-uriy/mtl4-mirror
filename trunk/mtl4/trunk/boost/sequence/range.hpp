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
# include <boost/sequence/distance.hpp>
# include <boost/sequence/intrinsic/operations_fwd.hpp>
# include <boost/sequence/detail/make_compressed_pair.hpp>
# include <boost/sequence/detail/is_mpl_integral_constant.hpp>

# include <boost/mpl/if.hpp>

# include <boost/type_traits/is_same.hpp>

namespace boost { namespace sequence { 

#ifdef BOOST_MSVC
# pragma warning(push)
# pragma warning(disable:4512)
#endif 

namespace range_
{
  struct accessor;
  
  struct not_stored
  {
      // Conversion from anything simplifies generic initialization
      template <class T> not_stored(T const&) {}
  };

  template <class Elements, class Begin, class End, class Size>
  class range
    : private compressed_pair<
          Begin
        , compressed_pair<
              End
            , compressed_pair<
                  Elements
                , Size
              >
          >
      >
  {
      friend struct accessor;
      
      typedef compressed_pair<
          Begin
        , compressed_pair<
              End
            , compressed_pair<
                  Elements
                , Size
              >
          >
      > base;

   public:
      typedef Elements elements;
      typedef Begin begin;
      typedef End end;
      typedef Size size;
      
      range(Elements const& m, Begin const& b, End const& e)
        : base(b, make_compressed_pair(e, detail::make_compressed_pair(m, Size())))
      {}

      range(Elements const& m, Begin const& b, End const& e, Size const& s)
        : base(b, detail::make_compressed_pair(e, detail::make_compressed_pair(m, s)))
      {}
  };

  struct accessor
  {
      template <class R>
      static typename R::base const& data(R const& r)
      { return r; }
  };

  // Identifies ranges for tag dispatch purposes
  struct tag {};
}


#ifdef BOOST_MSVC
# pragma warning(pop)
#endif 

using range_::range;

template <class Elements, class Begin, class End, class Size>
struct category<range<Elements,Begin,End, Size> >
  : mpl::eval_if<
        detail::is_mpl_integral_constant<Size>
      , mpl::identity<fixed_size::category>
      , mpl::if_<
            detail::o1_size_cursors<Begin,End>
          , o1_size_tag
          , sequence_tag
        >
    >
{};

namespace intrinsic
{
  template <class Elements, class Begin, class End, class Size>
  struct tag<range<Elements,Begin,End, Size> >
  {
      typedef range_::tag type;
  };

  template <class Range>
  struct operations<Range, range_::tag>
  {
      struct begin
      {
          typedef typename Range::begin type;
          type operator()(Range const& r) const
          {
              return range_::accessor::data(r).first();
          }
      };

      struct end
      {
          typedef typename Range::end type;
          type operator()(Range const& r) const
          {
              return range_::accessor::data(r).second().first();
          }
      };

      struct elements
      {
          typedef typename Range::elements type;
          type operator()(Range const& r) const
          {
              return range_::accessor::data(r).second().second().first();
          }
      };

      struct calculated_size
      {
          typedef typename intrinsic::distance<
              typename Range::begin, typename Range::end
          >::type type;
          
          type operator()(Range const& s) const
          {
              return sequence::distance( begin(s), end(s) );
          }
      };

      struct stored_size
      {
          typedef typename Range::size type;
          type operator()(Range const& r) const
          {
              return range_::accessor::data(r).second().second().second();
          }
      };
      
      struct size
        : mpl::if_<
              is_same<typename Range::size, range_::not_stored>
            , calculated_size
            , stored_size
          >::type
      {};
  };
}

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_RANGE_DWA200559_HPP
