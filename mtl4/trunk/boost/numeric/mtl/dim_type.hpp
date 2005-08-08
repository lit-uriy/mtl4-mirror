// $COPYRIGHT$

#ifndef MTL_DIM_TYPE_INCLUDE
#define MTL_DIM_TYPE_INCLUDE

#include <iostream>

namespace mtl {

  // Types for declaring matrix dimensions 
  // num_rows() and num_cols() return the number or rows and columns
  // is_static says whether it is declared at compile time or not

  // ... at compile time
  template <std::size_t Row, std::size_t Col>
  struct fix_dim_type 
  {
    typedef std::size_t size_type;
    
    std::size_t num_rows() const 
    {
      return Row;
    }
    std::size_t num_cols() const 
    {
      return Col;
    }

    // to check whether it is static
    static bool const is_static= true;

    typedef fix_dim_type<Col, Row> transpose_type;
    transpose_type transpose() const 
    { 
      return transpose_type(); 
    }
  };
  
  // ... at run time
  struct dim_type 
  {
    typedef std::size_t size_type;
    
    // some simple constructors
    dim_type() : r(0), c(0) {}
    dim_type(std::size_t rr, std::size_t cc) : r(rr), c(cc) {}
    dim_type(const dim_type& x) : r(x.r), c(x.c) {}
    dim_type(const std::pair<std::size_t, std::size_t>& x) : r(x.first), c(x.second) {}

    dim_type& operator=(const dim_type& x) 
    {
      r= x.r; c= x.c; return *this; 
    }
    std::size_t num_rows() const 
    {
      return r;
    }
    std::size_t num_cols() const {
      return c;
    }

    typedef dim_type transpose_type;
    transpose_type transpose() 
    { 
      return dim_type(c, r); 
    }
    static bool const is_static= false;
  protected:
    std::size_t r, c;
  };

  template <std::size_t R, std::size_t C>
  std::ostream& operator<< (std::ostream& stream, fix_dim_type<R, C>) 
  {
    return stream << R << 'x' << C; 
  }

  std::ostream& operator<< (std::ostream& stream, dim_type d) 
  {
    return stream << d.num_rows() << 'x' << d.num_cols(); 
  }

} // namespace mtl

#endif // MTL_DIM_TYPE_INCLUDE
