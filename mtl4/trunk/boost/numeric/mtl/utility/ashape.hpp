// $COPYRIGHT$

#ifndef MTL_ASHAPE_INCLUDE
#define MTL_ASHAPE_INCLUDE

#include <boost/type_traits.hpp>
#include <boost/mpl/if.hpp>

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>

namespace mtl { 

/// Namespace for algebraic shapes; used for sophisticated dispatching between operations
namespace ashape {


// Types (tags)
/// Scalar algebraic shape
struct scal {};
/// Row vector as algebraic shape
template <typename Value> struct rvec {};
/// Column vector as algebraic shape
template <typename Value> struct cvec {};
/// Matrix as algebraic shape
template <typename Value> struct mat {};
/// Undefined shape, e.g., for undefined results of operations
struct ndef {};

/// Meta-function for algebraic shape of T
/** Unknown types are treated like scalars. ashape of collections are template
    parameterized with ashape of their elements, e.g., ashape< matrix < vector <double> > >::type is
    mat< rvec <scal> > >. 
**/
template <typename T>
struct ashape
{
    typedef scal type;
};

/// Vectors must be distinguished between row and column vectors
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

template <typename Scaling, typename Coll>
struct ashape<matrix::scaled_view<Scaling, Coll> >
{
    typedef typename ashape<Coll>::type type;
};

template <typename Scaling, typename Coll>
struct ashape<vector::scaled_view<Scaling, Coll> >
{
    typedef typename ashape<Coll>::type type;
};

template <typename Coll>
struct ashape<matrix::conj_view<Coll> >
{
    typedef typename ashape<Coll>::type type;
};

template <typename Coll>
struct ashape<vector::conj_view<Coll> >
{
    typedef typename ashape<Coll>::type type;
};

template <typename Matrix>
struct ashape<transposed_view<Matrix> >
{
    typedef typename ashape<Matrix>::type type;
};

template <typename Matrix>
struct ashape<matrix::hermitian_view<Matrix> >
{
    typedef typename ashape<Matrix>::type type;
};

template <typename M1, typename M2>
struct ashape< matrix::mat_mat_plus_expr<M1, M2> >
{
    // M1 and M2 must have the same a-shape
    typedef typename ashape<M1>::type type;
};

template <typename M1, typename M2>
struct ashape< matrix::mat_mat_minus_expr<M1, M2> >
{
    // M1 and M2 must have the same a-shape
    typedef typename ashape<M1>::type type;
};

// =====================
// Shapes of products:
// =====================

// a) The result's shape
// b) Classify operation in terms of shape

// Operation types:

struct scal_scal_mult {};

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

 * only on outer level, forbidden for elements of collections

*/

// Results for elements of collections, i.e. scalar * matrix (vector) are excluded


/// Algebraic shape of multiplication's result when elements of collections are multiplied.
/** The types are the same as for multiplications of entire collections except that scalar *
    matrix (or vector) is excluded to avoid ambiguities. **/
template <typename Shape1, typename Shape2>
struct emult_shape
{
    typedef ndef type;
};

/// Type of operation when values of Shape1 and Shape2 are multiplied (so far only for elements of collections)
/** The types are the same as for multiplications of entire collections except that scalar *
    matrix (or vector) is excluded to avoid ambiguities. **/
template <typename Shape1, typename Shape2>
struct emult_op
{
    typedef ndef type;
};


// Scalar * scalar -> scalar
template <>
struct emult_shape<scal, scal>
{
    typedef scal type;
};

template <>
struct emult_op<scal, scal>
{
    typedef scal_scal_mult type;
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
