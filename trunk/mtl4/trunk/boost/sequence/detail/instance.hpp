// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_DETAIL_INSTANCE_DWA200559_HPP
# define BOOST_SEQUENCE_DETAIL_INSTANCE_DWA200559_HPP

# include <boost/sequence/detail/config.hpp>

namespace boost { namespace sequence { namespace detail { 

// Provides unique, default-constructed instances of T.
template <class T, int = 0>
struct instance
{
    static T& get()
    {
        static T x;
        return x;
    }
};

// BOOST_SEQUENCE_DECLARE_INSTANCE(type, name) --
//
//    Declares a

# if !BOOST_WORKAROUND(BOOST_GNUC_FULL_VERSION, <= 3003003)

// Declares an object "name" of type "type" that can be used in
// multiple translation units, without requiring a separate definition
// in a source file.
#  define BOOST_SEQUENCE_DECLARE_INSTANCE(type, name) \
    namespace { type const& name = sequence::detail::instance< type >::get(); }

# else

// This doesn't appear to cause problems, although strictly speaking
// it leads to ODR violations whenever a template that uses "name" is
// instantiated in multiple translation units.
#  define BOOST_SEQUENCE_DECLARE_INSTANCE(type, name) \
    namespace { type name; }

# endif

}}} // namespace boost::sequence::detail

#endif // BOOST_SEQUENCE_DETAIL_INSTANCE_DWA200559_HPP
