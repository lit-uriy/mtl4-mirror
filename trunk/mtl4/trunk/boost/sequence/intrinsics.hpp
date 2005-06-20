// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_INTRINSICS_DWA2005616_HPP
# define BOOST_SEQUENCE_INTRINSICS_DWA2005616_HPP

# include <boost/sequence/intrinsics_fwd.hpp>
# include <boost/sequence/fixed_size/intrinsics.hpp>
# include <boost/sequence/identity_property_map.hpp>
# include <boost/sequence/iterator_range_tag.hpp>
# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>
# include <boost/range/iterator.hpp>
# include <boost/range/const_iterator.hpp>

# include <boost/type_traits/is_array.hpp>
# include <boost/utility/enable_if.hpp>

namespace boost { namespace sequence {

template <class Sequence, class GetIterator>
struct iterator_range_intrinsics
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
        typedef typename GetIterator::type type;
        
        type operator()(Sequence& s) const
        {
            return type();
        }
    };
};

template <class Sequence>
struct intrinsics<Sequence, iterator_range_tag>
  : iterator_range_intrinsics<
        Sequence, range_iterator<Sequence>
    >
{};

template <class Sequence>
struct intrinsics<Sequence const, iterator_range_tag>
  : iterator_range_intrinsics<
        Sequence, range_const_iterator<Sequence>
    >
{};

namespace intrinsic
{
# define BOOST_SEQUENCE_INTRINSIC_OPERATION(name)   \
  template <class Sequence>                         \
  struct name                                       \
    : intrinsics<Sequence>::name                    \
  {};

BOOST_SEQUENCE_INTRINSIC_OPERATION(begin)
BOOST_SEQUENCE_INTRINSIC_OPERATION(end)
BOOST_SEQUENCE_INTRINSIC_OPERATION(elements)
      
  template <template <class> class Operation>
  struct function
  {
      template <class Sequence>
      typename Operation<Sequence>::type
      operator()(Sequence& s) const
      {
          return Operation<Sequence>()(s);
      }
      
      template <class Sequence>
      typename lazy_disable_if<is_array<Sequence>, Operation<Sequence const> >::type
      operator()(Sequence const& s) const
      {
          return Operation<Sequence const>()(s);
      }
  };
  
}
    
}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_INTRINSICS_DWA2005616_HPP
