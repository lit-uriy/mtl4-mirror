// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_ALGORITHM_FIXED_SIZE_COPY_DWA200559_HPP
# define BOOST_SEQUENCE_ALGORITHM_FIXED_SIZE_COPY_DWA200559_HPP

# include <cstddef>
# include <boost/sequence/advanced.hpp>
# include <boost/sequence/make_range.hpp>
# include <boost/sequence/next.hpp>
# include <boost/sequence/detail/make_compressed_pair.hpp>

namespace boost { namespace sequence { namespace algorithm { namespace fixed_size { 

//
// copy implementation
//
template <
    std::size_t N
  , class ReadablePropertyMap
  , class WritablePropertyMap
  , class ForwardCursorPair
>
compressed_pair<
    typename advanced<InCursor, mpl::size_t<N> >::type
  , typename advanced<OutCursor, mpl::size_t<N> >::type
>
copy(
    mpl::size_t<N>
  , ReadablePropertyMap const& in_elements
  , WritablePropertyMap const& out_elements
  , ForwardCursorPair const& cursors
)
{
    return copy(
        mpl::size_t<(N-N/2)>()
      , in_elements
      , out_elements
        
        copy(
            mpl::size_t<N/2>()
          , in_elements
          , out_elements
          , cursors
        ));
}
        
template <
    class ReadablePropertyMap, 
  , class WritablePropertyMap, class ForwardCursorPair
>
compressed_pair<
    typename successor<InCursor>::type
  , typename successor<OutCursor>::type
>
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

template <
    class ReadablePropertyMap, 
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
    
template <>
struct unrolled< copy_ >
{
    // Result type computer
    template <class Range1, class Range2>
    struct apply
    {
        typedef range<
            typename accessor<Range2>::type
          , typename advanced<
                typename begin_cursor<Range2>::type
              , typename extent<Range1>::type
            >::type
          , typename end_cursor<Range2>::type
        > type;
    };

    // Implementation
    template <class Range1, class Range2>
    static typename apply<Range1,Range2>::type
    execute(Range1 const& in, Range2& out)
    {
        return make_range(
            sequence::elements(out)
          , fixed_size::copy(
                typename extent<Range1>::type()

              , sequence::elements(in)
              , sequence::elements(out)

              , detail::make_compressed_pair(
                    sequence::begin(in)
                  , sequence::begin(out)
                )
            ).second()  // grab the out cursor
            
          , sequence::end(out)
        );
    }
};

}}}} // namespace boost::sequence::algorithm::fixed_size

#endif // BOOST_SEQUENCE_ALGORITHM_FIXED_SIZE_COPY_DWA200559_HPP
