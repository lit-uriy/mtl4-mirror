// $COPYRIGHT$

#ifndef MTL_ASHAPE_INCLUDE
#define MTL_ASHAPE_INCLUDE

#include <boost/numeric/mtl/mtl_fwd.hpp>

namespace mtl { namespace ashape {

// Algebraic shapes for more sophisticated dispatching between operations 

// Types (tags)
struct scal {};
template <typename Value> rvec {};
template <typename Value> cvec {};
template <typename Value> mat {};
// Not defined
struct ndef {};

// Unknown types are treated like scalars
template <typename T>
struct ashape
{
    typedef scal type;
};

// Vectors should be distinguished be
template <typename Value>
struct ashape<dense_vector<Value> >
{
    typedef cvec<typename ashape<Value>::type> type;
};
   
template <typename Value>
struct ashape<dense_row_vector<Value> >
{
    typedef rvec<typename ashape<Value>::type> type;
};
   
template <typename Value>
struct ashape<compressed2D<Value> >
{
    typedef mat<typename ashape<Value>::type> type;
};
   
template <typename Value>
struct ashape<dense2D<Value> >
{
    typedef mat<typename ashape<Value>::type> type;
};
   
template <typename Value, unsigned long Mask>
struct ashape<morton_dense<Value, Mask> >
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
    typedef err type;
};

template <>
struct emult_shape<scal, scal>
{
    typedef scal type;
};


template <>
struct emult_shape<cvec, rvec>
{
    typedef mat type;
};

template <>
struct emult_shape<rvec, cvec>
{
    typedef scal type;
};

template <>
struct emult_shape<rvec, mat>
{
    typedef rvec type;
};

template <>
struct emult_shape<mat, cvec>
{
    typedef cvec type;
};

template <>
struct emult_shape<mat, mat>
{
    typedef mat type;
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
