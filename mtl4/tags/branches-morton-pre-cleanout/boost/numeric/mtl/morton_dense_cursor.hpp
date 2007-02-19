// $COPYRIGHT$                                                                            

#ifndef MTL_MORTON_DENSE_CURSOR_INCLUDE
#define MTL_MORTON_DENSE_CURSOR_INCLUDE
    

namespace mtl {
    
// Morton Dense cursor
  template <class T, class OneD>
  class morton_dense_cursor {
  public:

    typedef T* iterator;
    typedef morton2D_iterator self;
    //typedef int distance_type;
    //typedef int difference_type;
    typedef int size_type;
    typedef std::random_access_iterator_tag iterator_category;

    typedef OneD*           pointer;
    typedef OneD            value_type;
    typedef OneD            reference;
    //typedef difference_type Distance;
    //typedef Iterator        iterator_type;
    
  protected:
    
    iterator start;
    size_type pos;
    size_type mIndex;
    size_type rows;
    size_type cols;
 
  public:
    
    inline size_type index() const { return pos; }

    inline morton2D_iterator() {}
    
    inline morton2D_iterator(iterator st, size_type ps, size_type mi, size_type rs,
      size_type cs) : start(st), pos(ps), mIndex(mi), rows(rs), cols(cs) {}
    
    inline morton2D_iterator(const self& x)
      : start(x.start), pos(x.pos), mIndex(x.mIndex), rows(x.rows), 
	cols(x.cols) {}

    inline self& operator=(const self& x) {
      start = x.start;
      pos = x.pos;
      mIndex = x.mIndex;
      rows = x.rows;
      cols = x.cols;
      return *this;
    }

    inline bool operator!=(const self& x) {
      return pos != x.pos;
    }

    inline size_type index() {
      return pos;
    }

    /*
  inline size_type row() {
    return rows;
  }

  inline size_type column() {
    return cols;
  }
    */

    inline reference operator*() { 
      // pos indicates the current row number
      // cols indicates the size of the 1-D row vector
      //return reference((T*)start + EvenDilate(pos), pos, cols);
      return reference((T*)start + mIndex, pos, cols);
    }

    inline self& operator++ () { ++pos; EvenInc(mIndex); return *this; }
    inline self operator++ (int) { self tmp = *this; ++pos; EvenInc(mIndex); return tmp; }

    inline self& operator+=(int n) { pos += n; mIndex = EvenDilate(pos); return *this; }
    inline self operator+(int n) const{
      self tmp = (*this);
      tmp += n;
      return tmp;
    }

  };

#endif
