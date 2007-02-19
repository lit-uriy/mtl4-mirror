// masked_int.h
// template class to support masked and unmasked integer
// for arbitrary mask

#ifndef MASKED_INT_H
#define MASKED_INT_H

#include <iostream>


template<class T>
static inline T inc(T i, T mask) {
  return ((i - mask) & mask);
}


template <class T, T mask>
class masked_int
{

  typedef masked_int self;

private:

  T value;               // value of the integer
  int n_bytes;           // number of bytes of the unmasked value of this type

public:

  static T mask_lut[sizeof(T)][256];
  static T unmask_lut[sizeof(T)][256];
  static T mask_piece[sizeof(T)];           // mask piece array
  static int mask_size[sizeof(T)];          // number of bits of each mask piece
  static int mask_shift_table[sizeof(T)];   // number of bits needed to shift 
                                            // for each mask table look-up
  static int unmask_shift_table[sizeof(T)]; // number of bits needed to shift for 
                                            // each unmask table look-up
  static int n_valid_table;                 // number of valid tables, == number 
                                            // of mask pieces


private:

  // initialize needed parameters
  void init() {
    n_bytes = sizeof(T);
    // calculate the number of valid table
    int n = count_n_ones(mask);
    if (n <= 8) n_valid_table = 1;
    else n_valid_table = (n%8 == 0 ? n/8 : n/8 + 1);           

    // set mask pieces 
    set_mask();    
  }


  // return the number of 1's in the mask
  int count_n_ones(T t) {
    int n_ones = 0;
    while(t) {
      if((t & 0x01) == 1) ++n_ones;
      t = t >>1;
    }
    return n_ones;
  };


  // return the number of valid bits in the mask
  int count_bits(T t) {
    int bits = 0;
    while(t) {
      ++bits;
      t = t >>1;
    }
    return bits;
  };


  // set mask pieces
  void set_mask() {

    // set the unmask shift table
    unmask_shift_table[0] = 0;
    T t_mask = mask;
    T tmp;
    int count;
    for (int i = 1; i < n_bytes; ++i) {
      tmp = t_mask & get_f_mask(8);
      count = count_n_ones(tmp);
      unmask_shift_table[i] = count + unmask_shift_table[i - 1];
      t_mask = t_mask >> (8*i);
    }

    mask_shift_table[0] = 0;  // don't need shift for the first table
    // if there is only 8 or less 1's in the mask,
    // only one table is needed
    if (n_valid_table == 1) {
      mask_piece[0] = mask; 
      mask_size[0] = count_bits(mask);
      return;
    }

    t_mask = mask;
    for (int i = 0; i < n_valid_table - 1; ++i) {
      int n_bits = 0;
      int n_ones = 0;
      T tmp = t_mask;
      while (n_ones < 8) {
	if ((t_mask & 0x01) == 1) ++n_ones;
	t_mask = t_mask >>1;
	++n_bits;
      } 
      // set the ith piece of mask, which must contains 8 1's
      mask_piece[i] = get_f_mask(n_bits) & tmp;

      // set the mask size table
      mask_size[i] = n_bits;

      // set shift table
      mask_shift_table[i + 1] = n_bits + mask_shift_table[i];
    }

    // set the last piece of mask, which may contain less than 8 1's
    // set the number of bits of the last mask
    mask_piece[n_valid_table - 1 ] = t_mask;    
    mask_size[n_valid_table - 1] = count_bits(t_mask);
  };


public:

  masked_int() : value(0) {
    init();
  }


  masked_int(const T val) : value(val & mask) {
  }


  T get_value() {
    return value;
  }


  void set_value(const T val) {
    value = val;
  }

 
  self operator+(const self d) {
    return self((value + (~mask) +  d.value) & mask);
  }


  self operator-(const self d) {
    return self((value - d.value) & mask);
  }


  // Increase and decrease operators (++, --)
  self& operator++ () {   // prefix ++
    //value = (value - mask) & mask;
    value = inc(value, mask);
    return *this;
  }

  self operator++ (int) {  // postfix ++
    self tmp = *this;
    ++(*this);
    return tmp;
  }

  self& operator-- () {   // prefix --
    value = (value - 1) & mask;
    return *this;
  }

  self operator-- (int) {  // postfix --
    self tmp = *this;
    --(*this);
    return tmp;
  }

  // plus 1 
  self& plus1() {
    return ++(*this);
  }

  // minus 1 
  self& minus1() {
    return --(*this);
  }



  // convert to masked integer
  T to_masked(T x) {
    T result = 0;
    for (int i = 0; i < n_valid_table; ++i)
      result += mask_lut[i][0xff & (x >> (8*i)) ];
    return result;
  }


  // convert to unmasked integer
  T to_unmasked(T x) {
    T result = 0;
    x &= mask;
    for (int i = 0; i < n_bytes; ++i)
      result += unmask_lut[i][0xff & (x >> (8*i)) ];
    return result;
  }


  // get mask of the style 0xfff...
  static T get_f_mask(int n_bits) {
    return (1 << n_bits) - 1;
  }

  struct compute_table {
    // compute mask table for the mask
    compute_table() {

      cout << "table computed! " << endl;

      // compute the mask table
      for (int j = 0; j < n_valid_table; ++j) {
	T f_mask = get_f_mask(mask_size[j]);
	T ii;
	int i;
	for (i = 0, ii = 0; i < 256; ++i, ii = inc(ii, mask_piece[j])) {
	  mask_lut[j][i] =  (ii & f_mask) << mask_shift_table[j]; // need to shift 
	}
      }

      // compute the unmask table
      T f_mask = get_f_mask(8);
      for (int j = 0; j < sizeof(T); ++j) {

	T t_mask = (mask >> (8*j)) & f_mask;
	T ii;
	int i;
	for(i = 0, ii = 0; ii < t_mask; ii = inc(ii, t_mask), ++i) {
	  unmask_lut[j][ii] =  i << unmask_shift_table[j];
	}
	// set the value for the last one
	unmask_lut[j][t_mask] =  i << unmask_shift_table[j];       
      }
    }
  };


  static void print_table() {
    for(int k = 0; k < n_valid_table; ++k) {
      cout << "\nMask table " << k << ":" << "\n\n";
      for (int i = 0; i < 32; i++) {
	for (int jj = 8*i; jj < 8*(i+1); jj++) {
	  cout << "  0x";
	  //	  cout.width(4*(k + 1)); 
	  cout.width(sizeof(T)*2); 
	  cout.fill('0');
	  cout << right << hex <<  mask_lut[k][jj];
	}
	cout << endl;
      }
    }

    for(int k = 0; k < sizeof(T); ++k) {
      cout << "\nUnmask table " << k << ":" << "\n\n";
        for (int i = 0; i < 32; i++) {
        for (int jj = 8*i; jj < 8*(i+1); jj++) {
	  cout << "  0x";
	  //	  cout.width(4 + k*2); 
	  cout.width(sizeof(T)*2); 
	  cout.fill('0');
	  cout << right << hex <<  unmask_lut[k][jj];
	}
	cout << endl;
	}
    }
  }

 public:

  static compute_table table_computed;

};


template <class T, T mask>
T masked_int<T, mask>::mask_lut[][256] = {};

template <class T, T mask>
T masked_int<T, mask>::unmask_lut[][256] = {};

template <class T, T mask>
T masked_int<T, mask>::mask_piece[] = {};

template <class T, T mask>
int masked_int<T, mask>::mask_size[] = {};

template <class T, T mask>
int masked_int<T, mask>::mask_shift_table[] = {};

template <class T, T mask>
int masked_int<T, mask>::unmask_shift_table[] = {};

template <class T, T mask>
int masked_int<T, mask>::n_valid_table = 0;

template <class T, T mask>
typename masked_int<T, mask>::compute_table masked_int<T, mask>::table_computed;


template <class T, T mask1, T mask2>
T convert(T value)
{  
  masked_int<T, mask1> a;
  masked_int<T, mask2> b;
  //  a.compute_table();  
  //  b.compute_table();
  T tmp = a.to_unmasked(value); 
  return b.to_masked(tmp);
} 

template <>
unsigned short int convert<unsigned short int, 0x5555, 0xaaaa>(unsigned short int value)
{
  return value << 1;
}

template <>
unsigned short int convert<unsigned short int, 0xaaaa, 0x5555>(unsigned short int value)
{
  return value >> 1;
}

template <>
unsigned int convert<unsigned int, 0x55555555, 0xaaaaaaaa>(unsigned int value)
{
  return value << 1;
}

template <>
unsigned int convert<unsigned int, 0xaaaaaaaa, 0x55555555>(unsigned int value)
{
  return value >> 1;
}

template <>
unsigned long convert<unsigned long, 0x55555555, 0xaaaaaaaa>(unsigned long value)
{
  return value << 1;
}

template <>
unsigned long convert<unsigned long, 0xaaaaaaaa, 0x55555555>(unsigned long value)
{
  return value >> 1;
}

#endif

