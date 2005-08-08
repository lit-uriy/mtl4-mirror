// $COPYRIGHT$

#ifndef MTL_PROPERTY_MAP_INCLUDE
#define MTL_PROPERTY_MAP_INCLUDE

namespace mtl { 

  namespace detail {

    // functor with matrix reference to access rows 
    template <class Matrix>
    struct indexer_row_ref
    {
      typedef Matrix                      matrix_type;
      typedef typename Matrix::key_type   key_type;
      indexer_row_ref(const matrix_type& ma) : ma(ma) {} 
      typename Matrix::size_type operator() (key_type key)
      {
	return ma.indexer.row(ma, key);
      }
      const matrix_type& ma;
    };
    
    template <class Matrix>
    struct indexer_col_ref
    {
      typedef Matrix                      matrix_type;
      typedef typename Matrix::key_type   key_type;
      indexer_col_ref(const matrix_type& ma) : ma(ma) {} 
      typename Matrix::size_type operator() (key_type key)
      {
	return ma.indexer.col(ma, key);
      }
      const matrix_type& ma;
    };

    // property map to read value if key is referring to value, e.g. pointer
    template <class Matrix>
    struct direct_const_value
    {
      direct_const_value(const Matrix&) {} // for compatibility
      typename Matrix::value_type operator() (typename Matrix::key_type key)
      {
	return *key;
      }
    };
    
    // same with writing
    template <class Matrix>
    struct direct_value : public direct_const_value<Matrix>
    {
      typedef typename Matrix::value_type value_type;
      direct_value(const Matrix& ma) : direct_const_value<Matrix>(ma) {} // for compatibility
      void operator() (typename Matrix::key_type key, value_type value)
      {
	* const_cast<value_type *>(key) = value;
      }
      typename Matrix::value_type operator() (typename Matrix::key_type key) // should be inherited
      {
	return *key;
      }
    };
    
  } // namespace detail

  namespace traits {
    
    template <class Matrix> struct row {};
    template <class Matrix> struct col {};
    template <class Matrix> struct const_value {};
    template <class Matrix> struct value {};

  } // namespace traits
}
#endif // MTL_PROPERTY_MAP_INCLUDE









// = old code to throw away if new code works


//   // will be removed probably and each matrix should provide its own method

//   template <class Matrix, class Key>
//   std::size_t row(const Matrix& ma, const Key& key) 
//   {
//     return ma.row(key); 
//   }

//   template <class Matrix, class Key>
//   std::size_t col(const Matrix& ma, const Key& key) 
//   {
//     return ma.col(key); 
//   }

//   template <class Matrix, class Key>
//   typename Matrix::value_type value(const Matrix& ma, const Key& key) 
//   {
//     return ma.value(key); 
//   }

//   template <class Matrix, class Key, class Val>
//   void value(Matrix& ma, const Key& key, const Val& val) 
//   {
//     ma.value(key, val); 
//   }

// #if 0
//   // better general solution relying on matrix's indexer object
//   template <class Matrix>
//   size_t row(const Matrix& ma, typename Matrix::key_type key)
//   {
//     return ma.indexer.row(ma, key);
//   }

//   template <class Matrix>
//   size_t row(const Matrix& ma, typename Matrix::key_type key)
//   {
//     return ma.indexer.row(ma, key);
//   }
// #endif
