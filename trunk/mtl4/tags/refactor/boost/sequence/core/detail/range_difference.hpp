// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_DETAIL_RANGE_DIFFERENCE_DWA20051129_HPP
# define BOOST_SEQUENCE_DETAIL_RANGE_DIFFERENCE_DWA20051129_HPP

# include <boost/sequence/minus.hpp>
# include <boost/sequence/intrinsic/sequence/size.hpp>
# include <boost/sequence/category.hpp>
# include <boost/sequence/core/operations.hpp>

namespace boost { namespace sequence { namespace detail {


template <class S1, class S2>
struct minimum_category
  : mpl::if_<
        is_convertible<
            typename category<S1>::type
          , typename category<S2>::type
        >
      , typename category<S2>::type
      , typename category<S1>::type
    >
{};

// No size stored by default
template <
    class InSequence, class OutSequence, class Begin
  , bool fixed = false
  , bool o1_size = false
>
struct range_difference_base
{
    typedef range<
        typename intrinsic::elements<OutSequence>::type
      , Begin
      , typename intrinsic::end<OutSequence>::type
    > type;

    type operator()(InSequence const& in, OutSequence& out, Begin const& b) const
    {
        return type( elements(out), b, end(out) );
    }
};

// Always store the size when it is fixed
template <class InSequence, class OutSequence, class Begin, bool o1_size>
struct range_difference_base<InSequence,OutSequence,Begin, true, o1_size>
{
    typedef range<
        typename intrinsic::elements<OutSequence>::type
      , Begin
      , typename intrinsic::end<OutSequence>::type
      , typename intrinsic::minus<
            typename intrinsic::size<OutSequence>::type
          , typename intrinsic::size<InSequence>::type
        >::type
    > type;

    type operator()(InSequence const& in, OutSequence& out, Begin const& b) const
    {
        return type( elements(out), b, end(out), minus(size(out), size(in)) );
    }
};

// When the size is O(1), decide whether to store based on whether the
// cursors can produce the size in O(1).
template <class InSequence, class OutSequence, class Begin>
struct range_difference_base<InSequence,OutSequence,Begin, false, true>
  : range_difference_base<
        InSequence
      , OutSequence
      , Begin
      , !o1_size_cursors<
            Begin
          , typename intrinsic::end<OutSequence>::type
        >::value
    >
{};
        
template <
    class InSequence, class OutSequence, class Begin,
    class MinCategory = typename minimum_category<InSequence,OutSequence>::type
>
struct range_difference
  : range_difference_base<
        InSequence,OutSequence,Begin
      , is_convertible<MinCategory,fixed_size::category>::value
      , is_convertible<MinCategory,o1_size_tag>::value
    >
{};

}}} // namespace boost::sequence::detail

#endif // BOOST_SEQUENCE_DETAIL_RANGE_DIFFERENCE_DWA20051129_HPP
