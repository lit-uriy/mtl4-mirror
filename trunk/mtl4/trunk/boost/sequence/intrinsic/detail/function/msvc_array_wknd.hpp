// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_DETAIL_MSVC_ARRAY_WKND_DWA2005815_HPP
# define BOOST_SEQUENCE_DETAIL_MSVC_ARRAY_WKND_DWA2005815_HPP

# include <boost/detail/workaround.hpp>

// BOOST_SEQUENCE_MSVC_ARRAY_WKND(args, gen_result_type) --
//
// A workaround for a VC++ bug that causes non-const arrays to bind to
// const references in preference to non-const references.
//
//   args: a Boost.Preprocessor SEQ of a function template's argument
//     types
//
//   gen_result_type: a nullary metafunction that computes the
//     function template's result type.
//
// Example:
//
//   template <class T>         // Should pick up non-const arrays
//   tyepename result<T&>::type
//   f(T& a0);                 
//
//   template <class T>
//   BOOST_SEQUENCE_MSVC_ARRAY_WKND( (T) , (result<T const&>) )
//   f(T const& a0);

# if !BOOST_WORKAROUND(_MSC_FULL_VER, BOOST_TESTED_AT(140050601))

#  define BOOST_SEQUENCE_MSVC_ARRAY_WKND(args, gen_result_type) \
    typename gen_result_type ::type

# else

#  include <boost/type_traits/is_array.hpp>
#  include <boost/utility/enable_if.hpp>
#  include <boost/mpl/or.hpp>

#  include <boost/sequence/intrinsic/detail/function/comma_protect.hpp>

#  include <boost/preprocessor/control/if.hpp>
#  include <boost/preprocessor/comparison/equal.hpp>
#  include <boost/preprocessor/seq/for_each.hpp>
#  include <boost/preprocessor/seq/for_each_i.hpp>
#  include <boost/preprocessor/seq/seq.hpp>
#  include <boost/preprocessor/seq/size.hpp>
#  include <boost/preprocessor/arithmetic/dec.hpp>
#  include <boost/preprocessor/cat.hpp>

#  define BOOST_SEQUENCE_CLOSE_ANGLE >
#  define BOOST_SEQUENCE_MSVC_ARRAY_WKND_TEST_1(elem) \
   boost::is_array<elem

#  define BOOST_SEQUENCE_MSVC_ARRAY_WKND_TEST_0(elem) \
   boost::mpl::or_< boost::is_array<elem> , 

#  define BOOST_SEQUENCE_MSVC_ARRAY_WKND_TEST(r, max, i, elem)      \
    BOOST_PP_CAT(                                                   \
        BOOST_SEQUENCE_MSVC_ARRAY_WKND_TEST_, BOOST_PP_EQUAL(max,i) \
    )(elem)

#  define BOOST_SEQUENCE_MSVC_ARRAY_WKND_CLOSE(r, x, elem) >

#  define BOOST_SEQUENCE_MSVC_ARRAY_WKND(args, gen_result_type)             \
   typename boost::lazy_disable_if<                                         \
       BOOST_PP_SEQ_FOR_EACH_I(                                             \
           BOOST_SEQUENCE_MSVC_ARRAY_WKND_TEST                              \
         , BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(args))                            \
         , args                                                             \
       )                                                                    \
       BOOST_PP_SEQ_FOR_EACH(BOOST_SEQUENCE_MSVC_ARRAY_WKND_CLOSE, ~, args) \
     , boost::sequence::detail::comma_protect<int gen_result_type>          \
   >::type

# endif 

#endif // BOOST_SEQUENCE_DETAIL_MSVC_ARRAY_WKND_DWA2005815_HPP
