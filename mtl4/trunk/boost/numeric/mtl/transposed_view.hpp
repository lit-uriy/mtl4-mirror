// $COPYRIGHT$

#ifndef MTL_TRANSPOSED_VIEW_INCLUDE
#define MTL_TRANSPOSED_VIEW_INCLUDE

#include <boost/numeric/mtl/base_types.hpp>

namespace mtl {


  template <class Matrix>
  class transposed_view {
    typedef transposed_view               self;
  public:	
    typedef Matrix                        other;
    typedef typename transposed_orientation<typename Matrix::orientation>::type orientation;
    typedef typename Matrix::index_type     index_type;
    typedef typename Matrix::value_type     value_type;
    typedef typename Matrix::pointer_type   pointer_type;
    typedef typename Matrix::key_type       key_type;
    typedef typename Matrix::el_cursor_type el_cursor_type;
    typedef std::pair<el_cursor_type, el_cursor_type> el_cursor_pair;

    transposed_view (other& ref) : ref(ref) {}
    
    el_cursor_pair elements() const 
    {
      return ref.elements();
    }

    value_type operator() (std::size_t r, std::size_t c) const
    { 
      return ref(c, r); 
    }

    std::size_t dim1() const 
    { 
      return ref.dim2(); 
    }
    std::size_t dim2() const 
    { 
      return ref.dim1(); 
    }
    
    std::size_t offset(const value_type* p) const 
    { 
      return ref.offset(p); 
    }
    pointer_type data_ref() const 
    {
      return ref.data_ref(); 
    }
    dim_type dim_ref() const 
    {
      return ref.dim_ref().transpose(); 
    }

    other& ref;
  };
  
  template <class Matrix>
  struct is_mtl_type<transposed_view<Matrix> > 
  {
    static bool const value= Matrix::value; 
  };

  template <class Matrix>
  struct which_matrix<transposed_view<Matrix> >
  {
    typedef typename Matrix::type type;
  };

  template <class Matrix>
  size_t row(const transposed_view<Matrix>& ma,
	     typename transposed_view<Matrix>::key_type key)
  {
    return col(ma.ref, key);
  }

  template <class Matrix>
  size_t col(const transposed_view<Matrix>& ma,
	     typename transposed_view<Matrix>::key_type key)
  {
    return row(ma.ref, key);
  }

  template <class Matrix>
  typename transposed_view<Matrix>::value_type 
  value(const transposed_view<Matrix>& ma,
	typename transposed_view<Matrix>::key_type key)
  {
    return value(ma.ref, key);
  }

  template <class Matrix>
  void value(const transposed_view<Matrix>& ma,
	     typename transposed_view<Matrix>::key_type key,
	     typename transposed_view<Matrix>::value_type v)
  {
    value(ma.ref, key, v);
  }

} // namespace mtl

#endif // MTL_TRANSPOSED_VIEW_INCLUDE








// = old code to throw away if new code works


//     std::size_t row(const key_type& key) const 
//     { 
//       return ref.col(key); 
//     }
//     std::size_t col(const key_type& key) const 
//     { 
//       return ref.row(key); 
//     }
//     value_type value(const key_type& key) const 
//     { 
//       return ref.value(key); 
//     }
//     // no overwriting of data at the moment
//     void value(const key_type& key, const value_type& value) 
//     { 
//       ref.value(key, value); 
//     }
//     value_type val_n(std::size_t offset) const 
//     { 
//       return ref.val_n(offset); 
//     }
    
