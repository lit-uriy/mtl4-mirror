// $COPYRIGHT$

#ifndef MTL_DIM_TYPE_INCLUDE
#define MTL_DIM_TYPE_INCLUDE

namespace mtl {

template <std::size_t R, std::size_t C>
class fix_dim_type {
public:
  typedef fix_dim_type<C, R> transpose_type;
  std::size_t rows() const {return R;}
  std::size_t cols() const {return C;}
  transpose_type transpose() const { return transpose_type(); }

  static const bool is_static= true;
};

class dim_type {
public:
  dim_type() : r(0), c(0) {}
  dim_type(std::size_t rr, std::size_t cc) : r(rr), c(cc) {}
  dim_type(const dim_type& x) : r(x.r), c(x.c) {}
  dim_type(const std::pair<std::size_t, std::size_t>& x) : r(x.first), c(x.second) {}

  dim_type& operator=(const dim_type& x) {
    r= x.r; c= x.c; return *this; }
  std::size_t rows() const {return r;}
  std::size_t cols() const {return c;}
  dim_type transpose() { return dim_type(c, r); }
  static const bool is_static= false;
protected:
  std::size_t r, c;
};


} // namespace mtl

#endif // MTL_DIM_TYPE_INCLUDE
