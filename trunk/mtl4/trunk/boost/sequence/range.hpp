// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_RANGE_DWA200559_HPP
# define BOOST_SEQUENCE_RANGE_DWA200559_HPP

# include <boost/sequence/range_fwd.hpp>
# include <boost/compressed_pair.hpp>
# include <boost/sequence/homogeneous.hpp>
# include <boost/sequence/category_fwd.hpp>
# include <boost/sequence/fixed_size/category.hpp>

namespace boost { namespace sequence { 

namespace range_
{
  template <class Elements, class Begin, class End>
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

  template <class Elements, class Begin, class End>
  Begin begin(range<Elements,Begin,End> const& r)
  {
      return r.first();
  }
  
  template <class Elements, class Begin, class End>
  End end(range<Elements,Begin,End> const& r)
  {
      return r.second().first();
  }

  template <class Elements, class Begin, class End>
  Elements const elements(range<Elements,Begin,End> const& r)
  {
      return r.second().second();
  }

  template <class Elements, class Begin, class End>
  Elements elements(range<Elements,Begin,End>& r)
  {
      return r.second().second();
  }
}

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

template <class Sequence> struct accessor;
template <class Elements, class Begin, class End>
struct accessor<range<Elements,Begin,End> const>
{
    typedef Elements const type;
};

template <class Elements, class Cursor>
struct category<range<Elements,Cursor,Cursor> > 
{
    typedef homogeneous type;
};

template <class Elements, class Cursor1, class Cursor2>
struct category<range<Elements,Cursor1,Cursor2> >
{
    typedef algorithm::fixed_size::category type;
};

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_RANGE_DWA200559_HPP
