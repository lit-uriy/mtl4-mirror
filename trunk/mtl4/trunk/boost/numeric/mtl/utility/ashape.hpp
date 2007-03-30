// $COPYRIGHT$

#ifndef MTL_ASHAPE_INCLUDE
#define MTL_ASHAPE_INCLUDE

#include <boost/type_traits.hpp>
#include <boost/mpl/if.hpp>

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>

namespace mtl { namespace ashape {

// Algebraic shapes for more sophisticated dispatching between operations 

// Types (tags)
struct scal {};
template <typename Value> struct rvec {};
template <typename Value> struct cvec {};
template <typename Value> struct mat {};
// Not defined
struct ndef {};

// Unknown types are treated like scalars
template <typename T>
struct ashape
{
    typedef scal type;
};

// Vectors should be distinguished be
template <typename Value, typename Parameters>
struct ashape<dense_vector<Value, Parameters> >
{
    typedef typename boost::mpl::if_<
	boost::is_same<typename Parameters::orientation, row_major>
      , rvec<typename ashape<Value>::type>
      , cvec<typename ashape<Value>::type>
    >::type type;
};

   
template <typename Value, typename Parameters>
struct ashape<compressed2D<Value, Parameters> >
{
    typedef mat<typename ashape<Value>::type> type;
};
   
template <typename Value, typename Parameters>
struct ashape<dense2D<Value, Parameters> >
{
    typedef mat<typename ashape<Value>::type> type;
};
   
template <typename Value, unsigned long Mask, typename Parameters>
struct ashape<morton_dense<Value, Mask, Parameters> >
{
    typedef mat<typename ashape<Value>::type> type;
};


// =====================
// Results of operations
// =====================

/* 
      s  cv  rv   m
-------------------
 s |  s  cv* rv*  m*
cv | cv*  x   m   x
rv | rv*  s   x  rv
 m |  m* cv   x   m 

 * only on outer level, for elements forbidden

*/

// Results for elements of collections, i.e. scalar * matrix (vector) are excluded

template <typename Shape1, typename Shape2>
struct emult_shape
{
    typedef ndef type;
};

template <>
struct emult_shape<scal, scal>
{
    typedef scal type;
};


template <typename Value1, typename Value2>
struct emult_shape<cvec<Value1>, rvec<Value2> >
{
    typedef mat<typename emult_shape<Value1, Value2>::type> type;
};

template <typename Value1, typename Value2>
struct emult_shape<rvec<Value1>, cvec<Value2> >
{
    typedef typename emult_shape<Value1, Value2>::type type;
};

template <typename Value1, typename Value2>
struct emult_shape<rvec<Value1>, mat<Value2> >
{
    typedef rvec<typename emult_shape<Value1, Value2>::type> type;
};

template <typename Value1, typename Value2>
struct emult_shape<mat<Value1>, cvec<Value2> >
{
    typedef cvec<typename emult_shape<Value1, Value2>::type> type;
};

template <typename Value1, typename Value2>
struct emult_shape<mat<Value1>, mat<Value2> >
{
    typedef mat<typename emult_shape<Value1, Value2>::type> type;
};

// Results for entire collections, i.e. scalar * matrix (vector) are allowed

template  <typename Shape1, typename Shape2>
struct mult_shape
    : public emult_shape<Shape1, Shape2>
{};

template <typename Shape2>
struct mult_shape<scal, Shape2>
{
    typedef Shape2 type;
};

template <typename Shape1>
struct mult_shape<Shape1, scal>
{
    typedef Shape1 type;
};

// Arbitration
template <>
struct mult_shape<scal, scal>
{
    typedef scal type;
};


}} // namespace mtl::ashape

#endif // MTL_ASHAPE_INCLUDE
