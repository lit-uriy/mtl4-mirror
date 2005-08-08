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
  template <class ELT> //, class Offset>
  struct dense_el_cursor : public detail::base_cursor<const ELT*> 
  {
    typedef ELT                           value_type;
    typedef const value_type*             pointer_type; // ?
    typedef detail::base_cursor<const ELT*> super;

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
  template <class ELT, class Orientation= row_major, class Index= index::c_index,
	    class Dimension= mtl::dim_type>
  class dense2D : public detail::base_matrix<ELT, Orientation, Dimension> {
    typedef detail::base_matrix<ELT, Orientation, Dimension>      super;
    typedef dense2D                       self;
  public:	
    typedef Orientation                   orientation;
    typedef Index                         index_type;
    typedef ELT                           value_type;
    typedef const value_type*             pointer_type;
    typedef pointer_type                  key_type;
    typedef std::size_t                   size_type;
    typedef Dimension                     dim_type;
    typedef dense_el_cursor<ELT>          el_cursor_type;  
    typedef std::pair<el_cursor_type, el_cursor_type> el_cursor_pair;
    typedef dense2D_indexer               indexer_type;

  protected:
    void set_nnz()
    {
      nnz = dim.num_rows() * dim.num_cols();
    }

    void allocate() 
    {
      set_nnz();
      super::allocate();
    }
  public:
    // if compile time matrix size allocate memory
    dense2D() : super() 
    {
      if (dim_type::is_static) allocate();
    }

    // only sets dimensions, only for run-time dim
    explicit dense2D(mtl::dim_type d) : super(d) {} 

    // sets dimensions and pointer to external data
    explicit dense2D(mtl::dim_type d, value_type* a) : super(d, a) 
    { 
      set_nnz();
    }

    // allocates memory and sets all values to 'value'
    explicit dense2D(mtl::dim_type d, value_type value) : super(d) 
    {
      allocate();
      std::fill(data, data+nnz, value);
    }

    // copies values from iterators
    template <class InputIterator>
    explicit dense2D(mtl::dim_type d, InputIterator first, InputIterator last) : super(d) 
    {
      allocate();
      std::copy(first, last, data);
   }

    // same constructors for compile time matrix size
    // sets dimensions and pointer to external data
    explicit dense2D(value_type* a) : super(a) 
    { 
//       BOOST_ASSERT((dim_type::is_static));
    }

    // allocates memory and sets all values to 'value'
    explicit dense2D(value_type value) 
    {
//       BOOST_ASSERT((dim_type::is_static));
      allocate();
      std::fill(data, data+nnz, value);
    }

    // copies values from iterators
    template <class InputIterator>
    explicit dense2D(InputIterator first, InputIterator last) 
    {
//       BOOST_ASSERT((dim_type::is_static));
      allocate();
      std::copy(first, last, data);
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


  template <class ELT, class Orientation, class Index, class Dimension>
  struct is_mtl_type<dense2D<ELT, Orientation, Index, Dimension> > 
  {
    static bool const value= true; 
  };

  // define corresponding type without all template parameters
  template <class ELT, class Orientation, class Index, class Dimension>
  struct which_matrix<dense2D<ELT, Orientation, Index, Dimension> > 
  {
    typedef dense2D_tag type;
  };

  namespace traits {
    template <class ELT, class Orientation, class Index, class Dimension>
    struct row<dense2D<ELT, Orientation, Index, Dimension> >
    {
      typedef mtl::detail::indexer_row_ref<dense2D<ELT, Orientation, Index, Dimension> > type;
    };

    template <class ELT, class Orientation, class Index, class Dimension>
    struct col<dense2D<ELT, Orientation, Index, Dimension> >
    {
      typedef mtl::detail::indexer_col_ref<dense2D<ELT, Orientation, Index, Dimension> > type;
    };

    template <class ELT, class Orientation, class Index, class Dimension>
    struct const_value<dense2D<ELT, Orientation, Index, Dimension> >
    {
      typedef mtl::detail::direct_const_value<dense2D<ELT, Orientation, Index, Dimension> > type;
    };

    template <class ELT, class Orientation, class Index, class Dimension>
    struct value<dense2D<ELT, Orientation, Index, Dimension> >
    {
      typedef mtl::detail::direct_value<dense2D<ELT, Orientation, Index, Dimension> > type;
    };

  } // namespace traits
    

  template <class ELT, class Orientation, class Index, class Dimension>
  inline typename traits::row<dense2D<ELT, Orientation, Index, Dimension> >::type
  row(const dense2D<ELT, Orientation, Index, Dimension>& ma) 
  {
    return typename traits::row<dense2D<ELT, Orientation, Index, Dimension> >::type(ma);
  }

  template <class ELT, class Orientation, class Index, class Dimension>
  inline typename traits::col<dense2D<ELT, Orientation, Index, Dimension> >::type
  col(const dense2D<ELT, Orientation, Index, Dimension>& ma)
  {
    return typename traits::col<dense2D<ELT, Orientation, Index, Dimension> >::type(ma);
  }

  template <class ELT, class Orientation, class Index, class Dimension>
  inline typename traits::const_value<dense2D<ELT, Orientation, Index, Dimension> >::type
  const_value(const dense2D<ELT, Orientation, Index, Dimension>& ma)
  {
    return typename traits::const_value<dense2D<ELT, Orientation, Index, Dimension> >::type(ma);
  }

  template <class ELT, class Orientation, class Index, class Dimension>
  inline typename traits::value<dense2D<ELT, Orientation, Index, Dimension> >::type
  value(const dense2D<ELT, Orientation, Index, Dimension>& ma)
  {
    return typename traits::value<dense2D<ELT, Orientation, Index, Dimension> >::type(ma);
  }

} // namespace mtl

#endif // MTL_DENSE2D_INCLUDE














// = old code to throw away if new code works

//   template <class Matrix>
//   struct dense2D_indexer 
//   {
//     typedef Matrix                        matrix_type;
//     typedef typename Matrix::orientation  orientation;
//     typedef typename Matrix::index_type   index_type;
//     typedef typename Matrix::key_type     key_type;

//   private:
//     std::size_t _offset(const matrix_type& ma, std::size_t r, std::size_t c, row_major) const {
//       return r * ma.dim2() + c; }
//     std::size_t _offset(const matrix_type& ma, std::size_t r, std::size_t c, col_major) const {
//       return c * ma.dim2() + r; }
    
//     std::size_t _row(const matrix_type& ma, const key_type& key, row_major) const {
//       return ma.offset(key) / ma.dim2(); }
//     std::size_t _row(const matrix_type& ma, const key_type& key, col_major) const {
//       return ma.offset(key) % ma.dim2(); }
    
//     std::size_t _col(const matrix_type& ma, const key_type& key, row_major) const {
//       return ma.offset(key) % ma.dim2(); }
//     std::size_t _col(const matrix_type& ma, const key_type& key, col_major) const {
//       return ma.offset(key) / ma.dim2(); }

//   public:
//     // dealing with fortran indices here (to do it only once) and orientation above
//     std::size_t operator() (const matrix_type& ma, std::size_t r, std::size_t c) const {
//       return _offset(ma, idec(r, ind()), idec(c, ind()), orientation()); }
    
//     std::size_t row(const matrix_type& ma, const key_type& key) const {
//       return iinc( _row(ma, key, orientation()), ind() ); }
    
//     std::size_t col(const matrix_type& ma, const key_type& key) const {
//       return iinc( _col(ma, key, orientation()), ind() ); }
//   };

//   size_t row(const key_type& key) const {
//     return indexer.row(*this, key); }
//   size_t col(const key_type& key) const {
//     return indexer.col(*this, key); }
//   value_type value(const key_type& key) const {
//     return *key; }
//   void value(const key_type& key, const value_type& value) {
//     * const_cast<value_type *>(key)= value; }

// template <class ELT, class Orientation= row_major, class Indexing= c_index>
// struct dense2D : public dense2D_impl<ELT, dense2D_indexer<Orientation, Indexing> > {};



// // declare as fortran indexed if so
// template <class ELT, class Orientation, class Index, class Dimension>
// struct indexing<dense2D<ELT, Orientation, Index, Dimension> > {
//   typedef Index type; };
// // should be done automatically

//   template <class ELT, class Orientation, class Index, class Dimension>
//   size_t row(const dense2D<ELT, Orientation, Index, Dimension>& ma,
// 	     typename dense2D<ELT, Orientation, Index, Dimension>::key_type key)
//   {
//     dense2D_indexer indexer;
//     return indexer.row(ma, key);
//   }

//   template <class ELT, class Orientation, class Index, class Dimension>
//   size_t col(const dense2D<ELT, Orientation, Index, Dimension>& ma,
// 	     typename dense2D<ELT, Orientation, Index, Dimension>::key_type key)
//   {
//     dense2D_indexer indexer;
//     return indexer.col(ma, key);
//   }

//   template <class ELT, class Orientation, class Index, class Dimension>
//   ELT value(const dense2D<ELT, Orientation, Index, Dimension>& ma,
// 	    typename dense2D<ELT, Orientation, Index, Dimension>::key_type key)
//   {
//     return *key; 
//   }

//   template <class ELT, class Orientation, class Index, class Dimension>
//   void value(dense2D<ELT, Orientation, Index, Dimension>& ma,
// 	     typename dense2D<ELT, Orientation, Index, Dimension>::key_type key,
// 	     ELT v)
//   {
//     * const_cast<ELT *>(key) = v;
//   }

//   // functor with matrix reference to access rows (could be hidden in detail ???)
//   template <class ELT, class Orientation, class Index, class Dimension>
//   struct dense2D_row_ref
//   {
//     typedef dense2D<ELT, Orientation, Index, Dimension> matrix_type;
//     typedef typename matrix_type::key_type              key_type;
//     dense2D_row_ref(const matrix_type& ma) : ma(ma) {}
//     operator() (key_type key)
//     {
//       return ma.indexer.row(ma, key);
//     }
//     const matrix_type& ma;
//   };

  
