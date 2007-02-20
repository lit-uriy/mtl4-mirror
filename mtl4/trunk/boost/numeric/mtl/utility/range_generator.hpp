// $COPYRIGHT$

#ifndef MTL_RANGE_GENERATOR_INCLUDE
#define MTL_RANGE_GENERATOR_INCLUDE

#include <boost/numeric/mtl/detail/range_generator.hpp>
#include <boost/numeric/mtl/utility/complexity.hpp>

namespace mtl {

namespace traits
{
    // Functor for generating begin and end cursors over a collection
    // Thus functor must contain begin and end member functions which are used by free functions
    // The cursor type must be defined as 'typedef xxx type;'
    // complexity characterizes the run time of a traveral, cf. utility/complexity.hpp
    //   complexity can be used to dispatch between different traversals depending on algorithm
    //   and on collection (corr. subset represented by cursor)
    // level indicates the maximal level of nesting
    //   - level 0: there is no traversal of this Tag for this collection
    //   - level 1: cursor refers directly to elements
    //   - level 2: cursor iterates over sets of elements or is only an intermediate cursor
    //              cursor e.g. over rows, 
    //              its generated ranges are level 1 and iterate over elements
    //   - level 3: cursor over sets of sets of elements,
    //              its generated ranges are level 2 or 1 depending on the tag used on the cursor
    //   - level 4: for instance blocked matrix -> level 4: block rows -> level 3: block elements
    //                -> level 2: regular rows -> level 1: matrix elements
    //              if an element cursor range was generated from the block element then the nesting 
    //                would be only 3 (since the last two levels collapse)
    // Cursors of level > 1 represent subsets of a collection and thus, it is only logical that
    // there must be range generators for these subset, which are applied on the cursor.
    template <typename Tag, typename Collection>
    struct range_generator
    {
	typedef complexity_classes::infinite  complexity;
	static int const             level = 0;
	// specializations must contain the following members
	// typedef xxx               type;
	// type begin() { ... }
	// type end()   { ... }
    };
} // namespace traits



// Returns begin cursor over the Collection or a subset of the Collection
// Form of traversal depends on Tag, cf utility/glas_tag.hpp
// On nested traversals, cursors of level > 1 must provide at least one range generator
template <class Tag, class Collection>
typename traits::range_generator<Tag, Collection>::type 
inline begin(Collection const& c)
{
  return traits::range_generator<Tag, Collection>().begin(c);
}

// Corresponding end cursor
template <class Tag, class Collection>
typename traits::range_generator<Tag, Collection>::type 
inline end(Collection const& c)
{
  return traits::range_generator<Tag, Collection>().end(c);
}


} // namespace mtl

#endif // MTL_RANGE_GENERATOR_INCLUDE
