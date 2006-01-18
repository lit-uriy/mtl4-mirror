// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_ALGORITHM_UNROLLED_COPY_DWA200559_HPP
# define BOOST_SEQUENCE_ALGORITHM_UNROLLED_COPY_DWA200559_HPP

# include <cstddef>
# include <boost/sequence/make_range.hpp>
# include <boost/sequence/range.hpp>
# include <boost/sequence/intrinsic/cursor/next.hpp>
# include <boost/sequence/intrinsic/sequence/size.hpp>
# include <boost/sequence/intrinsic/sequence/elements.hpp>
# include <boost/sequence/intrinsic/sequence/begin.hpp>
# include <boost/sequence/intrinsic/sequence/end.hpp>

# include <boost/sequence/algorithm/unrolled/dispatch.hpp>
# include <boost/sequence/core/detail/make_compressed_pair.hpp>
# include <boost/sequence/algorithm/copy_fwd.hpp>

# include <boost/sequence/core/detail/range_difference.hpp>

# include <boost/sequence/core/advance.hpp>

# include <boost/mpl/size_t.hpp>

namespace boost { namespace sequence { namespace algorithm {

//
// copy implementation
//

// A metafunction that returns a pair whose first and second types
// equal to advanced versions of the first and second types in the
// given ForwardCursorPair
template <class ForwardCursorPair, std::size_t N>
struct advance_cursor_pair
{
    typedef mpl::size_t<N> distance;
    typedef typename intrinsic::advance<
        typename ForwardCursorPair::first_type
      , distance
    >::type c1;
    
    typedef typename intrinsic::advance<
        typename ForwardCursorPair::second_type
      , distance
    >::type c2;
    
    typedef compressed_pair<c1,c2> type;
};


// Here is the unrolled copy implementation.
//
// We are not yet attempting to limit compile-time unrolling for
// large ranges; there is surely a point where large fixed-size ranges
// should be copied by unrolled subsequence copies.
//
// We are also not trying to dispatch to machine intrinsics, yet.
template <>
struct unrolled<id::copy>
{
    // Result type computer.  This metafunction is called "apply" so
    // that unrolled<id::copy> will be an MPL metafunction class.
    template <class Range1, class Range2>
    struct apply
      : detail::range_difference<
            Range1,
            Range2,
            typename intrinsic::begin<Range2>::type
        >
    {};

    // Implementation
    template <class Range1, class Range2>
    static typename apply<Range1,Range2>::type
    execute(Range1 const& in, Range2& out)
    {
        typedef typename intrinsic::size<Range1>::type length;
        
        return apply<Range1,Range2>()
        (
            in, out
                
          , unrolled::copy(
                mpl::size_t<length::value>()
              , sequence::elements(in)
              , sequence::elements(out)

              , detail::make_compressed_pair(
                    sequence::begin(in)
                  , sequence::begin(out)
                )
            ).second()  // grab the out cursor
        );
    }

    // Recursion terminating case
    template <
        class ReadablePropertyMap
      , class WritablePropertyMap
      , class ForwardCursorPair
    >
    static ForwardCursorPair
    copy(
        mpl::size_t<0>
      , ReadablePropertyMap const& in_elements
      , WritablePropertyMap const& out_elements
      , ForwardCursorPair const& cursors
    )
    {
        return cursors;
    }

    // Copy a single element
    template <
        class ReadablePropertyMap
      , class WritablePropertyMap
      , class ForwardCursorPair
    >
    static typename advance_cursor_pair<ForwardCursorPair,1>::type
    copy(
        mpl::size_t<1>
      , ReadablePropertyMap const& in_elements
      , WritablePropertyMap const& out_elements
      , ForwardCursorPair const& cursors
    )
    {
        out_elements(*cursors.second(), in_elements(*cursors.first()));

        return detail::make_compressed_pair(
            sequence::next(cursors.first())
          , sequence::next(cursors.second())
        );
    }

    // Copy Length elements where Length > 1
    template <
        std::size_t Length
      , class ReadablePropertyMap
      , class WritablePropertyMap
      , class ForwardCursorPair
    >
    static typename advance_cursor_pair<ForwardCursorPair,Length>::type
    copy(
        mpl::size_t<Length>
      , ReadablePropertyMap const& in_elements
      , WritablePropertyMap const& out_elements
      , ForwardCursorPair const& cursors
    )
    {
        static std::size_t const length_of_1st_half = Length/2;
        static std::size_t const length_of_2nd_half = Length - length_of_1st_half;

        // Recursively copy the second half and then the 1st half.
        return copy(
            mpl::size_t<length_of_2nd_half>()
          , in_elements
          , out_elements

          , copy(
              mpl::size_t<length_of_1st_half>()
            , in_elements
            , out_elements
            , cursors
          ));
    }
};

}}} // namespace boost::sequence::algorithm

#endif // BOOST_SEQUENCE_ALGORITHM_UNROLLED_COPY_DWA200559_HPP
