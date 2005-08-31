// $COPYRIGHT$

#ifndef MTL_DENSE2D_INCLUDE
#define MTL_DENSE2D_INCLUDE

#include <algorithm>
#include <boost/numeric/mtl/detail/base_cursor.hpp>
#include <boost/numeric/mtl/detail/base_matrix.hpp>
#include <boost/numeric/mtl/dim_type.hpp>
#include <boost/numeric/mtl/base_types.hpp>
#include <boost/numeric/mtl/index.hpp>
#include <boost/numeric/mtl/property_map.hpp>

namespace mtl {

using std::size_t;

// cursor over every element
template <class Elt> //, class Offset>
  struct dense_el_cursor : public detail::base_cursor<const Elt*> 
  {
    typedef Elt                           value_type;
    typedef const value_type*             pointer_type; // ?
    typedef detail::base_cursor<const Elt*> super;

    dense_el_cursor () {} 
    dense_el_cursor (pointer_type me_) : super(me_) {}
  };


  struct dense2D_indexer 
  {
  private:
    // helpers for public functions
    size_t _offset(size_t dim2, size_t r, size_t c, row_major) const 
    {
      return r * dim2 + c; 
    }
    size_t _offset(size_t dim2, size_t r, size_t c, col_major) const 
    {
      return c * dim2 + r; 
    }
    
    size_t _row(size_t offset, size_t dim2, row_major) const 
    {
      return offset / dim2; 
    }
    size_t _row(size_t offset, size_t dim2, col_major) const 
    {
      return offset % dim2;
    }
    
    size_t _col(size_t offset, size_t dim2, row_major) const 
    {
      return offset % dim2;
    }
    size_t _col(size_t offset, size_t dim2, col_major) const 
    {
      return offset / dim2; 
    }

 public:
    template <class Matrix>
    size_t operator() (const Matrix& ma, size_t r, size_t c) const
    {
      // convert into c indices
      typename Matrix::index_type my_index;
      size_t my_r= index::change_from(my_index, r);
      size_t my_c= index::change_from(my_index, c);
      return _offset(ma.dim2(), my_r, my_c, typename Matrix::orientation());
    }

    template <class Matrix>
    size_t row(const Matrix& ma, typename Matrix::key_type key) const
    {
      // row with c-index for my orientation
      size_t r= _row(ma.offset(key), ma.dim2(), typename Matrix::orientation());
      return index::change_to(typename Matrix::index_type(), r);
    }

    template <class Matrix>
    size_t col(const Matrix& ma, typename Matrix::key_type key) const 
    {
      // column with c-index for my orientation
      size_t c= _col(ma.offset(key), ma.dim2(), typename Matrix::orientation());
      return index::change_to(typename Matrix::index_type(), c);
    }
  };

  
  // M and N as template parameters might be considered later
  template <class Elt, class Orientation= row_major, class Index= index::c_index,
	    class Dimensions= mtl::non_fixed::dimensions>
  class dense2D : public detail::base_matrix<Elt, Orientation, Dimensions> {
    typedef detail::base_matrix<Elt, Orientation, Dimensions>      super;
    typedef dense2D                       self;
  public:	
    typedef Orientation                   orientation;
    typedef Index                         index_type;
    typedef Elt                           value_type;
    typedef const value_type*             pointer_type;
    typedef pointer_type                  key_type;
    typedef std::size_t                   size_type;
    typedef Dimensions                     dim_type;
    typedef dense_el_cursor<Elt>          el_cursor_type;  
    typedef std::pair<el_cursor_type, el_cursor_type> el_cursor_pair;
    typedef dense2D_indexer               indexer_type;

  protected:
    void set_nnz()
    {
      nnz = dim.num_rows() * dim.num_cols();
    }

  public:
    // Allocate memory
    void allocate() 
    {
      set_nnz();
      super::allocate();
    }

    // if compile time matrix size allocate memory
    dense2D() 
      : super() 
    {
        if (dim_type::is_static) allocate();
    }

    // only sets dimensions, only for run-time dimensions
    explicit dense2D(mtl::non_fixed::dimensions d) 
      : super(d) 
    {
        allocate();
    } 

    // sets dimensions and pointer to external data
    explicit dense2D(mtl::non_fixed::dimensions d, value_type* a) 
      : super(d, a) 
    { 
        set_nnz();
    }

    // same constructor for compile time matrix size
    // sets dimensions and pointer to external data
    explicit dense2D(value_type* a) : super(a) 
    { 
        // BOOST_ASSERT((dim_type::is_static));
    }

    // friend class indexer_type; should work without friend declaration

    // begin and end cursors to iterate over all matrix elements
    el_cursor_pair elements() const
    {
      return std::make_pair(data_ref(), data_ref() + nnz);
    }

    // old style, better use value property map
    value_type operator() (size_t r, size_t c) const 
    {
      size_t offset= indexer(*this, r, c);
      return data[offset];
      // return data[indexer(*this, r, c)]; 
    }

    indexer_type  indexer;
  }; // dense2D


  template <class Elt, class Orientation, class Index, class Dimensions>
  struct is_mtl_type<dense2D<Elt, Orientation, Index, Dimensions> > 
  {
    static bool const value= true; 
  };

  // define corresponding type without all template parameters
  template <class Elt, class Orientation, class Index, class Dimensions>
  struct matrix_category<dense2D<Elt, Orientation, Index, Dimensions> > 
  {
    typedef dense2D_tag type;
  };

  namespace traits 
  {
    template <class Elt, class Orientation, class Index, class Dimensions>
    struct row<dense2D<Elt, Orientation, Index, Dimensions> >
    {
      typedef mtl::detail::indexer_row_ref<dense2D<Elt, Orientation, Index, Dimensions> > type;
    };

    template <class Elt, class Orientation, class Index, class Dimensions>
    struct col<dense2D<Elt, Orientation, Index, Dimensions> >
    {
      typedef mtl::detail::indexer_col_ref<dense2D<Elt, Orientation, Index, Dimensions> > type;
    };

    template <class Elt, class Orientation, class Index, class Dimensions>
    struct const_value<dense2D<Elt, Orientation, Index, Dimensions> >
    {
      typedef mtl::detail::direct_const_value<dense2D<Elt, Orientation, Index, Dimensions> > type;
    };

    template <class Elt, class Orientation, class Index, class Dimensions>
    struct value<dense2D<Elt, Orientation, Index, Dimensions> >
    {
      typedef mtl::detail::direct_value<dense2D<Elt, Orientation, Index, Dimensions> > type;
    };

  } // namespace traits
    

  template <class Elt, class Orientation, class Index, class Dimensions>
  inline typename traits::row<dense2D<Elt, Orientation, Index, Dimensions> >::type
  row(const dense2D<Elt, Orientation, Index, Dimensions>& ma) 
  {
    return typename traits::row<dense2D<Elt, Orientation, Index, Dimensions> >::type(ma);
  }

  template <class Elt, class Orientation, class Index, class Dimensions>
  inline typename traits::col<dense2D<Elt, Orientation, Index, Dimensions> >::type
  col(const dense2D<Elt, Orientation, Index, Dimensions>& ma)
  {
    return typename traits::col<dense2D<Elt, Orientation, Index, Dimensions> >::type(ma);
  }

  template <class Elt, class Orientation, class Index, class Dimensions>
  inline typename traits::const_value<dense2D<Elt, Orientation, Index, Dimensions> >::type
  const_value(const dense2D<Elt, Orientation, Index, Dimensions>& ma)
  {
    return typename traits::const_value<dense2D<Elt, Orientation, Index, Dimensions> >::type(ma);
  }

  template <class Elt, class Orientation, class Index, class Dimensions>
  inline typename traits::value<dense2D<Elt, Orientation, Index, Dimensions> >::type
  value(const dense2D<Elt, Orientation, Index, Dimensions>& ma)
  {
    return typename traits::value<dense2D<Elt, Orientation, Index, Dimensions> >::type(ma);
  }

} // namespace mtl

#endif // MTL_DENSE2D_INCLUDE




