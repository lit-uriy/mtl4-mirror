// $COPYRIGHT$

#ifndef MTL_PROPERTY_MAP_INCLUDE
#define MTL_PROPERTY_MAP_INCLUDE

namespace mtl { namespace detail {

// functor with matrix reference to access rows 
template <class Matrix> struct indexer_row_ref
{
    typedef Matrix                      matrix_type;
    typedef typename Matrix::key_type   key_type;
    indexer_row_ref(const matrix_type& ma) : ma(ma) {} 
    
    typename Matrix::size_type operator() (key_type key) const
    {
	return ma.indexer.row(ma, key);
    }
    const matrix_type& ma;
};

// functor with matrix reference to access rows 
template <class Matrix> struct row_in_key
{
    typedef Matrix                      matrix_type;
    typedef typename Matrix::key_type   key_type;
    row_in_key(const matrix_type&) {} 
    
    typename Matrix::size_type operator() (key_type const& key) const
    {
	return key.row();
    }
};

    
template <class Matrix> struct indexer_col_ref
{
    typedef Matrix                      matrix_type;
    typedef typename Matrix::key_type   key_type;
    indexer_col_ref(const matrix_type& ma) : ma(ma) {} 
    
    typename Matrix::size_type operator() (key_type key) const
    {
	return ma.indexer.col(ma, key);
    }
    const matrix_type& ma;
};

// functor with matrix reference to access cols 
template <class Matrix> struct col_in_key
{
    typedef Matrix                      matrix_type;
    typedef typename Matrix::key_type   key_type;
    col_in_key(const matrix_type&) {} 
    
    typename Matrix::size_type operator() (key_type const& key) const
    {
	return key.col();
    }
};


// property map to read value if key is referring to value, e.g. pointer
template <class Matrix> struct direct_const_value
{
    direct_const_value(const Matrix&) {} // for compatibility
    typename Matrix::value_type operator() (typename Matrix::key_type key) const
    {
	return *key;
    }
};
    
// same with writing
template <class Matrix> struct direct_value 
  : public direct_const_value<Matrix>
{
    typedef typename Matrix::value_type value_type;

    direct_value(const Matrix& ma) 
      : direct_const_value<Matrix>(ma) 
    {} // for compatibility

    void operator() (typename Matrix::key_type key, value_type value)
    {
	* const_cast<value_type *>(key) = value;
    }

    // should be inherited
    typename Matrix::value_type operator() (typename Matrix::key_type key) const
    {
	return *key;
    }
};
    
} // namespace detail

namespace traits 
{    
    template <class Matrix> struct row {};
    template <class Matrix> struct col {};
    template <class Matrix> struct const_value {};
    template <class Matrix> struct value {};

} // namespace traits

}
#endif // MTL_PROPERTY_MAP_INCLUDE


