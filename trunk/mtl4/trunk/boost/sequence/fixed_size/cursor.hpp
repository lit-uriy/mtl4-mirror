// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef CURSOR_DWA2005330_HPP
# define CURSOR_DWA2005330_HPP

# include <boost/sequence/detail/is_mpl_integral_constant.hpp>
# include <boost/utility/enable_if.hpp>
# include <boost/type_traits/is_class.hpp>
# include <boost/mpl/integral_c.hpp>
# include <boost/mpl/bool_.hpp>
# include <boost/typeof/typeof.hpp>
# include <cstddef>

namespace boost {
namespace sequence {
namespace fixed_size { 

  namespace cursor_ // namespace for ADL protection.
  {
    // A cursor that stores a compile-time index.
    template <std::size_t N>
    struct cursor
    {
        static std::size_t const value = N;
        operator std::size_t() const { return N; }
        cursor operator*() const { return *this; }
        typedef cursor type;
    };

    // The difference of two cursors is an integral constant
    template <std::size_t N1, std::size_t N2>
    inline mpl::integral_c<std::ptrdiff_t,(N1-N2)>
    operator-(cursor<N1>, cursor<N2>)
    {
        return mpl::int_<(N1-N2)>();
    }

    // Relational operators returning mpl::bool_ constants.
  # define BOOST_SEQUENCE_fixed_size_cursor_relational(op) \
    template <std::size_t N1, std::size_t N2>              \
    inline mpl::bool_<(N1 op N2)>                          \
    operator op(cursor<N1>,cursor<N2>)                     \
    {                                                      \
        return mpl::bool_<(N1 op N2)>();                   \
    }

    BOOST_SEQUENCE_fixed_size_cursor_relational(==)
    BOOST_SEQUENCE_fixed_size_cursor_relational(!=)
    BOOST_SEQUENCE_fixed_size_cursor_relational(<)
    BOOST_SEQUENCE_fixed_size_cursor_relational(<=)
    BOOST_SEQUENCE_fixed_size_cursor_relational(>=)
    BOOST_SEQUENCE_fixed_size_cursor_relational(>)

  # undef BOOST_SEQUENCE_fixed_size_cursor_relational

    // Increment and decrement
    template <std::size_t N>
    inline cursor<(N+1)> next(cursor<N>)
    {
        return cursor<(N+1)>();
    }

    template <std::size_t N>
    inline cursor<(N-1)> prev(cursor<N>)
    {
        return cursor<(N-1)>();
    }

    //
    // Random access interactions with integral constants -- these
    // operations preserve the integrity of the cursor's position as a
    // compile-time constant.
    //

    // helper metafunctions
    template <std::size_t N, class T>
    struct add
    {
        typedef cursor<(N + T::value)> type;
    };

    template <std::size_t N, class T>
    struct subtract
    {
        typedef cursor<(N - T::value)> type;
    };


    template <std::size_t N, class T>
    typename lazy_enable_if<
        is_mpl_integral_constant<T>::type
      , cursor_::add<N,T>
    >::type
    inline operator+(cursor<N>, T)
    {
        return cursor<(N+T::value)>();
    }

    template <std::size_t N, class T>
    typename lazy_enable_if<
        is_mpl_integral_constant<T>::type
      , cursor_::add<N,T>
    >::type
    inline operator+(T,cursor<N>)
    {
        return cursor<(N+T::value)>();
    }

    template <std::size_t N, class T>
    typename lazy_enable_if<
        is_mpl_integral_constant<T>::type
      , cursor_::subtract<N,T>
    >::type
    inline operator-(cursor<N>, T)
    {
        return cursor<(N-T::value)>();
    }

    //
    // Random access interactions with runtime integers.
    //

    template <std::size_t N, class T>
    typename enable_if_c<
        boost::is_integral<T>::type
      , BOOST_TYPEOF_TPL(sequence::detail::make<T>()+N)
    >::type
    inline operator+(cursor<N>, T x)
    {
        return x + N;
    }

    template <std::size_t N, class T>
    typename enable_if_c<
        boost::is_integral<T>::type
      , BOOST_TYPEOF_TPL(sequence::detail::make<T>()+N)
    >::type
    inline operator+(T x,cursor<N>)
    {
        return x + N;
    }

    template <std::size_t N, class T>
    typename enable_if_c<
        boost::is_integral<T>::type
      , BOOST_TYPEOF_TPL(N - sequence::detail::make<T>())
    >::type
    inline operator-(cursor<N>, T x)
    {
        return N - x;
    }
  } // namespace cursor_

  using cursor_::cursor;

} // namespace fixed_size

template <class Cursor> struct successor;
template <class Cursor> struct predecessor;

template <std::size_t N>
struct successor<fixed_size::cursor<N> >
{
    typedef fixed_size::cursor<N+1> type;
};

template <std::size_t N>
struct predecessor<fixed_size::cursor<N> >
{
    typedef fixed_size::cursor<N-1> type;
};

template <std::size_t N>
struct dereferenced<fixed_size::cursor<N> >
{
    typedef cursor<N> type;
};

template <std::size_t N>
struct dereferenced<fixed_size::cursor<N> >
{
    typedef cursor<N> type;
};

template <std::size_t N1, std::size_t N2>
struct difference<fixed_size::cursor<N1>, fixed_size::cursor<N2> >
{
    typedef mpl::integral_c<std::ptrdiff_t,(N1-N2)> type;
};

}} // namespace boost::sequence::fixed_size

#endif // CURSOR_DWA2005330_HPP
