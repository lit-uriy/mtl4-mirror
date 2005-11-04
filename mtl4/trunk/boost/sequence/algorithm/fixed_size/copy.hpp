// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_ALGORITHM_FIXED_SIZE_COPY_DWA200559_HPP
# define BOOST_SEQUENCE_ALGORITHM_FIXED_SIZE_COPY_DWA200559_HPP

# include <cstddef>
# include <boost/sequence/make_range.hpp>
# include <boost/sequence/range.hpp>
# include <boost/sequence/next.hpp>
# include <boost/sequence/homogenize.hpp>
# include <boost/sequence/extent.hpp>
# include <boost/sequence/elements.hpp>
# include <boost/sequence/begin.hpp>
# include <boost/sequence/end.hpp>

# include <boost/sequence/algorithm/fixed_size/unrolled.hpp>
# include <boost/sequence/detail/make_compressed_pair.hpp>
# include <boost/sequence/algorithm/copy_fwd.hpp>

# include <boost/mpl/size_t.hpp>

namespace boost { namespace sequence { namespace algorithm { namespace fixed_size { 

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
    
    typedef compressed_pair<const c1,const c2> type;
};

// Recursion terminating case
template <
    class ReadablePropertyMap
  , class WritablePropertyMap
  , class ForwardCursorPair
>
ForwardCursorPair
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
typename advance_cursor_pair<ForwardCursorPair,1>::type
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
typename advance_cursor_pair<ForwardCursorPair,Length>::type
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
    return fixed_size::copy(
        mpl::size_t<length_of_2nd_half>()
      , in_elements
      , out_elements
        
      , fixed_size::copy(
          mpl::size_t<length_of_1st_half>()
        , in_elements
        , out_elements
        , cursors
      ));
}

// Here is the unrolled copy implementation.
//
// We are not yet specializing for the case where the sequence is
// homogeneous, i.e. where any position in the sequence _can_ be
// represented with a single (runtime) cursor type.  In those cases we
// can save instantiations by calculating the length at compile-time
// but passing different runtime cursors to indicate the subsequences
// to be copied.
//
// We are also not yet attempting to limit compile-time unrolling for
// large ranges; there is surely a point where large fixed-size ranges
// should be copied by unrolled subsequence copies.
//
// We are also not trying to dispatch to machine intrinsics, yet.
template <>
struct unrolled< id::copy, false >
{
    // Result type computer.  This metafunction is called "apply" so
    // that unrolled<id::copy> will be an MPL metafunction class.
    template <class Range1, class Range2>
    struct apply
    {
        typedef Range2 r;
        typedef typename intrinsic::begin<Range2>::type start;
        typedef typename intrinsic::end<Range2>::type fini;
        
        typedef range<
            typename intrinsic::elements<Range2>::type
          , typename intrinsic::advance<
                start
              , typename extent<Range1>::type
            >::type
          , typename intrinsic::end<Range2>::type
        > type;
    };

    template <class Advancer, class Start, class Length, class Finish>
    static typename Advancer::type
    select_result(Advancer a, Start s, Length l, Finish)
    {
        return a(s,l);
    }
    
    template <class Advancer, class Start, class Length>
    static typename Advancer::type
    select_result(Advancer, Start s, Length l, typename Advancer::type f)
    {
        return f;
    }
    
    // Implementation
    template <class Range1, class Range2>
    static typename apply<Range1,Range2>::type
    execute(Range1 const& in, Range2& out)
    {
        typedef typename extent<Range1>::type length;
        typedef intrinsic::advance<typename apply<Range1,Range2>::start, length> advancer;
            
        return make_range(
            sequence::elements(out)
          , select_result(
                advancer()
              , sequence::begin(out)
              , length()
                
              , fixed_size::copy(
                    mpl::size_t<length::value>()
                  , sequence::elements(sequence::homogenize(in))
                  , sequence::elements(sequence::homogenize(out))

                  , detail::make_compressed_pair(
                        sequence::begin(sequence::homogenize(in))
                      , sequence::begin(sequence::homogenize(out))
                    )
                ).second()  // grab the out cursor
            )
            
          , sequence::end(out)
        );
    }
};

template <>
struct unrolled< id::copy, true >
  : unrolled< id::copy, false >
{
    // Implementation
    template <class Range1, class Range2>
    static typename apply<Range1,Range2>::type
    execute(Range1 const& in, Range2& out)
    {
        typedef typename extent<Range1>::type length;
            
        return make_range(
            sequence::elements(out)
          , sequence::advance(
              , sequence::begin(out)
              , length()
            )
          , sequence::end(out)
        );
    }
}

}}}} // namespace boost::sequence::algorithm::fixed_size

#endif // BOOST_SEQUENCE_ALGORITHM_FIXED_SIZE_COPY_DWA200559_HPP
