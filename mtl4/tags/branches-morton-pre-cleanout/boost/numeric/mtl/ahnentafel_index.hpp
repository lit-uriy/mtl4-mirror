////////////////////////////////////////////////
// AhnenIndex.hpp
//
// class to support Ahnentafel index for Morton
// matrix. All boundary checking is done in the 
// Morton matrix class.
// 
// 
// Jiahu Deng
// Last updated:  01/29/2006
////////////////////////////////////////////////


#ifndef AHNENTAFEL_INDEX_INCLUDE
#define AHNENTAFEL_INDEX_INCLUDE

#include <boost/numeric/mtl/morton_index.hpp>

class AhnenIndex
{

  typedef AhnenIndex self_type;

public:
  
  // constructors
  AhnenIndex() { index_ = 3; }
  AhnenIndex(const self_type& a):index_(a.index_) { }
  AhnenIndex(const int index):index_(index) { }

  // assignment operator
  self_type& operator=(const self_type& a) {
    index_ = a.index_;
    return *this;
  }
 
  // destructors
  ~AhnenIndex() { };

  // functions to access index
  int getIndex() const { return index_; };
  void setIndex(int index) { index_ = index; };

  // functions to access children and parents of current index
  // self_type nwChild() const { return AhnenIndex(4 * index_); }
  // self_type swChild() const { return AhnenIndex(4 * index_ + 1); }
  // self_type neChild() const { return AhnenIndex(4 * index_ + 2); }
  // self_type seChild() const { return AhnenIndex(4 * index_ + 3); }
  // self_type parent() const { return AhnenIndex(static_cast<int>(index_ / 4)); }

  // functions to access the indice of children and parents 
  int getNwChildIndex() const { return 4 * index_; }
  int getSwChildIndex() const { return 4 * index_ + 1; }
  int getNeChildIndex() const { return 4 * index_ + 2; }
  int getSeChildIndex() const { return 4 * index_ + 3;}
  int getParentIndex() const { return static_cast<int>(index_ / 4); }

  // functions to access children and parents of current index
  self_type nwChild() const { return AhnenIndex(getNwChildIndex()); }
  self_type swChild() const { return AhnenIndex(getSwChildIndex()); }
  self_type neChild() const { return AhnenIndex(getNeChildIndex()); }
  self_type seChild() const { return AhnenIndex(getSeChildIndex()); }
  self_type parent() const { return AhnenIndex(getParentIndex()); }

  // return the ahnentafel index of the block at the same level as 
  // current block that sits on the main diagonal horizontal from 
  // current block
  self_type diag() const { return AhnenIndex(3 *(index_ & EvenBits)); }
  


private:
  
  int index_;     // Ahnentafel index
};

#endif
