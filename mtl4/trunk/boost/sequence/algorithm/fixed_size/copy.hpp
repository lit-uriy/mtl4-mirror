// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_ALGORITHM_FIXED_SIZE_COPY_DWA200559_HPP
# define BOOST_SEQUENCE_ALGORITHM_FIXED_SIZE_COPY_DWA200559_HPP

# include <cstddef>
# include <boost/sequence/advanced.hpp>
# include <boost/sequence/make_range.hpp>
# include <boost/sequence/range.hpp>
# include <boost/sequence/next.hpp>
# include <boost/sequence/algorithm/fixed_size/unrolled.hpp>
# include <boost/sequence/detail/make_compressed_pair.hpp>
# include <boost/sequence/algorithm/copy_fwd.hpp>
# include <boost/sequence/extent.hpp>
# include <boost/sequence/elements.hpp>
# include <boost/sequence/begin.hpp>
# include <boost/sequence/end.hpp>
# include <boost/mpl/size_t.hpp>

namespace boost { namespace sequence { namespace algorithm { namespace fixed_size { 

//
// copy implementation
//

template <class ForwardCursorPair, std::size_t N>
struct advance_cursor_pair
{
    typedef mpl::size_t<N> distance;
    typedef typename advanced<
        typename ForwardCursorPair::first_type
      , distance
    >::type c1;
    
    typedef typename advanced<
        typename ForwardCursorPair::second_type
      , distance
    >::type c2;
    
    typedef compressed_pair<const c1,const c2> type;
};

template <std::size_t Length>
struct copy
{
    static std::size_t const h1 = Length/2;
    static std::size_t const h2 = Length - h1;
    template <
        class ReadablePropertyMap
      , class WritablePropertyMap
      , class ForwardCursorPair
    >
    typename advance_cursor_pair<ForwardCursorPair,Length>::type
# if 0
    compressed_pair<
        typename advanced<typename ForwardCursorPair::first_type, mpl::size_t<Length> >::type
      , typename advanced<typename ForwardCursorPair::second_type, mpl::size_t<Length> >::type
        >
# endif 
    operator()(
        ReadablePropertyMap const& in_elements
      , WritablePropertyMap const& out_elements
      , ForwardCursorPair const& cursors
    ) const
    {
    
        return fixed_size::copy<h2>()(
            in_elements
          , out_elements
        
          , fixed_size::copy<h1>()(
              in_elements
            , out_elements
            , cursors
          ));
    }
};


template <>
struct copy<0>
{
    template <
        class ReadablePropertyMap
      , class WritablePropertyMap
      , class ForwardCursorPair
    >
    ForwardCursorPair
    operator()(
        ReadablePropertyMap const& in_elements
      , WritablePropertyMap const& out_elements
      , ForwardCursorPair const& cursors
    ) const
    {
        return cursors;
    }
};
        
template <>
struct copy<1>
{
    template <
        class ReadablePropertyMap
      , class WritablePropertyMap
      , class ForwardCursorPair
    >
    typename advance_cursor_pair<ForwardCursorPair,1>::type
# if 0
    compressed_pair<
        typename successor<typename ForwardCursorPair::first_type>::type
      , typename successor<typename ForwardCursorPair::second_type>::type
        >
# endif 
    operator()(
        ReadablePropertyMap const& in_elements
      , WritablePropertyMap const& out_elements
      , ForwardCursorPair const& cursors
    ) const
    {
        out_elements(*cursors.second(), in_elements(*cursors.first()));
    
        return detail::make_compressed_pair(
            sequence::next(cursors.first())
          , sequence::next(cursors.second())
        );
    }
};

# if 0
int f()
{
    sequence::next(sequence::fixed_size::cursor<0>());
    copy<2>()(
        identity_property_map()
      , identity_property_map()
      , detail::make_compressed_pair(
          sequence::fixed_size::cursor<0>()
        , sequence::fixed_size::cursor<0>()
      )
    );
}
# endif 

template <>
struct unrolled< copy_ >
{
    // Result type computer
    template <class Range1, class Range2>
    struct apply
    {
        typedef Range2 r;
        typedef typename intrinsic::begin<Range2>::type start;
        typedef typename intrinsic::end<Range2>::type fini;
        
        typedef range<
            typename intrinsic::elements<Range2>::type
          , typename advanced<
                start
              , typename extent<Range1>::type
            >::type
          , typename intrinsic::end<Range2>::type
        > type;
    };

    // Implementation
    template <class Range1, class Range2>
    static typename apply<Range1,Range2>::type
    execute(Range1 const& in, Range2& out)
    {
        typedef typename extent<Range1>::type length;

        return make_range(
            sequence::elements(out)
          , fixed_size::copy<length::value>()(
                sequence::elements(in)
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
