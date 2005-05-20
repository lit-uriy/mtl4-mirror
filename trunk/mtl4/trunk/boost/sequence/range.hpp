// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_RANGE_DWA200559_HPP
# define BOOST_SEQUENCE_RANGE_DWA200559_HPP

# include <boost/compressed_pair.hpp>

namespace boost { namespace sequence { 

namespace range_
{
  template <class Elements, class Begin, class End = Begin>
  struct range
    : private compressed_pair<
          Begin
        , compressed_pair<
              End
            , Elements
          >
      >
  {
      typedef compressed_pair<
          Begin
        , compressed_pair<
              End
            , Elements
          >
      > base;
      
      range(Elements const& m, Begin const& b, End const& e)
        : base(b, compressed_pair<End,Elements>(e,m))
      {}
  };
}

using range_::range;

template <class Sequence> struct begin_cursor;
template <class Elements, class Begin, class End>
struct begin_cursor<range<Elements,Begin,End> >
{
    typedef Begin type;
};

template <class Sequence> struct end_cursor;
template <class Elements, class Begin, class End>
struct end_cursor<range<Elements,Begin,End> >
{
    typedef End type;
};

template <class Sequence> struct accessor;
template <class Elements, class Begin, class End>
struct accessor<range<Elements,Begin,End> >
{
    typedef Elements type;
};

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_RANGE_DWA200559_HPP
