// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_INTRINSIC_OPERATIONS_DWA2005616_HPP
# define BOOST_SEQUENCE_INTRINSIC_OPERATIONS_DWA2005616_HPP

# include <boost/sequence/operations/operations_fwd.hpp>

# include <boost/sequence/core/property_map/identity_property_map.hpp>
# include <boost/sequence/core/iterator_range_tag.hpp>

# include <boost/sequence/core/detail/is_mpl_integral_constant.hpp>

# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>
# include <boost/range/iterator.hpp>
# include <boost/range/const_iterator.hpp>
# include <boost/range/size.hpp>

# include <boost/iterator/is_lvalue_iterator.hpp>

# include <boost/type_traits/remove_cv.hpp>
# include <boost/type_traits/is_convertible.hpp>

# include <boost/mpl/size_t.hpp>
# include <boost/array.hpp>

namespace boost { namespace sequence { namespace intrinsic {

// The implementation of the intrinsics specialization that applies to
// models of SinglePassRange (see the Boost Range library
// documentation at http://www.boost.org/libs/range/doc/range.html).
// The first argument is the range type and the second argument is a
// nullary metafunction that returns the iterator type to use as the
// Range's cursor.
//
// Note: the need for the 2nd argument may disappear when the Range
// library is changed so that range_iterator<R const>::type is the
// same as range_const_iterator<R>::type.
template <class Sequence, class GetIterator>
struct iterator_range_operations
{
    struct begin
    {
        typedef typename GetIterator::type type;
        
        type operator()(Sequence& s) const
        {
            return boost::begin(s);
        }
    };
        
    struct end
    {
        typedef typename GetIterator::type type;
        
        type operator()(Sequence& s) const
        {
            return boost::end(s);
        }
    };
        
    struct elements
    {
        typedef identity_property_map type;
        
        type operator()(Sequence const&) const
        {
            return type();
        }
    };

    struct size
    {
        typedef typename boost::range_size<
            typename remove_cv<Sequence>::type
        >::type type;

        type operator()(Sequence& s) const
        {
            
            return boost::size(s);
        }
    };
};

// Intrinsics specializations for iterator ranges.
template <class Sequence>
struct operations<Sequence, iterator_range_tag>
  : iterator_range_operations<
# if 1
        Sequence, range_iterator<Sequence>
# else
        // This appears to be code for proxy support.  We will need
        // something like this, unless Thorsten fixes his library
        // first.
        typename mpl::if_<
            mpl::or_<
                is_convertible<
                    typename iterator_category<
                        typename range_iterator<Sequence>::type
                    >::type
                  , std::output_iterator_tag
                >
              , is_non_const_lvalue_iterator<
                    typename range_iterator<Sequence>::type
                >
            >
          , Sequence
          , Sequence const
        >::type
      , range_result_iterator<Sequence>
# endif 
    >
{};

template <class Sequence>
struct operations<Sequence const, intrinsic::iterator_range_tag>
  : iterator_range_operations<
        Sequence const, range_const_iterator<Sequence>
    >
{};


// The default implementation of each intrinsic function object type
// is inherited from the corresponding member of
// operations<Sequence>.  You can of course specialize begin<S>,
// end<S>, and elements<S>, individually, but specializing
// operations<> usually more convenient.
template <class Sequence>
struct begin : operations<Sequence>::begin {};
  
template <class Sequence>
struct end : operations<Sequence>::end {};
  
template <class Sequence>
struct elements : operations<Sequence>::elements {};
  
template <class Sequence>
struct size : operations<Sequence>::size {};

template <class Sequence>
struct size<Sequence const>
  : size<Sequence>
{};

template <class Sequence>
struct size<Sequence&>
  : size<Sequence>
{};

template <class T, std::size_t N>
struct size<T[N]>
{
    typedef mpl::size_t<N> type;
    type operator()(T const (&)[N]) { return type(); }
};

template <class T, std::size_t N>
struct size<T const[N]> : size<T[N]> {};

template <class T, std::size_t N>
struct size<boost::array<T,N> >
{
    typedef mpl::size_t<N> type;
    type operator()(boost::array<T,N> const&) { return type(); }
};

template <class T, std::size_t N>
struct size<array<T,N> const> : size<array<T,N> > {};


template <class Cursor>
struct next
{
    typedef Cursor type;
    Cursor operator()(Cursor x) const
    {
        return ++x;
    }
};

template <class Cursor>
struct next<Cursor const>
  : next<Cursor>
{};
  
template <class Cursor>
struct prev
{
    typedef Cursor type;
    Cursor operator()(Cursor x) const
    {
        return --x;
    }
};
  
template <class Cursor>
struct prev<Cursor const>
  : prev<Cursor>
{};
  
template <class Cursor1, class Cursor2>
struct equal
{
    typedef bool type;
      
    type operator()(Cursor1 const& c1, Cursor2 const& c2) const
    {
        return c1 == c2;
    }
};

template <class Cursor1, class Cursor2>
struct equal<Cursor1 const, Cursor2 const>
  : equal<Cursor1,Cursor2>
{};
  
template <class Cursor1, class Cursor2>
struct equal<Cursor1 const, Cursor2>
  : equal<Cursor1,Cursor2>
{};
  
template <class Cursor1, class Cursor2>
struct equal<Cursor1, Cursor2 const>
  : equal<Cursor1,Cursor2>
{};
  

}}} // namespace boost::sequence::intrinsic

#endif // BOOST_SEQUENCE_INTRINSIC_OPERATIONS_DWA2005616_HPP
