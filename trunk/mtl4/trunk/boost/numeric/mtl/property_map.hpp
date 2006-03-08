// $COPYRIGHT$

#ifndef MTL_PROPERTY_MAP_INCLUDE
#define MTL_PROPERTY_MAP_INCLUDE

#include <boost/numeric/mtl/property_map.hpp>

namespace mtl { namespace traits 
{    
    template <class Matrix> struct row {};
    template <class Matrix> struct col {};
    template <class Matrix> struct const_value {};
    template <class Matrix> struct value {};

}} // namespace mtl::traits






namespace mtl { namespace detail {

// functor with matrix reference to access rows 
template <class Matrix> struct indexer_row_ref
{
    typedef Matrix                      matrix_type;
    typedef typename Matrix::key_type   key_type;
    indexer_row_ref(const matrix_type& ma) : ma(ma) {} 
    
    typename Matrix::size_type operator() (key_type const& key) const
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
    
    typename Matrix::size_type operator() (key_type const& key) const
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

template <typename Matrix>
struct const_value_from_other
{
    typedef typename Matrix::other     other;
    typedef typename other::key_type   key_type;
    typedef typename other::value_type value_type;

    explicit const_value_from_other(Matrix const& matrix) 
	: its_const_value(matrix.ref) {}

    value_type operator() (key_type const& key) const
    {
	return its_const_value(key);
    }

  protected:
    typename traits::const_value<other>::type  its_const_value;
};

template <typename Matrix>
struct value_from_other
{
    typedef typename Matrix::other     other;
    typedef typename other::key_type   key_type;
    typedef typename other::value_type value_type;

    explicit value_from_other(Matrix const& matrix) 
	: its_value(matrix.ref) {}

    value_type operator() (key_type const& key) const
    {
	return its_value(key);
    }

    void operator() (key_type const& key, value_type value) const
    {
	its_value(key, value);
    }

  protected:
    typename traits::value<other>::type  its_value;
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

    // May be to be replaced by inserter
    void operator() (typename Matrix::key_type const& key, value_type value)
    {
	* const_cast<value_type *>(key) = value;
    }

    // should be inherited
    typename Matrix::value_type operator() (typename Matrix::key_type const& key) const
    {
	return *key;
    }
};


    
template <class Matrix> struct matrix_const_value_ref
{
    typedef Matrix                      matrix_type;
    typedef typename Matrix::key_type   key_type;
    matrix_const_value_ref(const matrix_type& ma) : ma(ma) {} 
    
    typename Matrix::value_type operator() (key_type const& key) const
    {
	return ma(key);
    }
    const matrix_type& ma;
};

template <class Matrix> struct matrix_value_ref
{
    typedef Matrix                      matrix_type;
    typedef typename Matrix::key_type   key_type;
    typedef typename Matrix::value_type value_type;
    matrix_value_ref(matrix_type& ma) : ma(ma) {} 
    
    typename Matrix::value_type operator() (key_type const& key) const
    {
	return ma(key);
    }

    // Much better with inserters
    void operator() (typename Matrix::key_type const& key, value_type const& value)
    {
	ma(key, value);
    }

    matrix_type& ma;
};

}} // namespace mtl::detail


#endif // MTL_PROPERTY_MAP_INCLUDE


