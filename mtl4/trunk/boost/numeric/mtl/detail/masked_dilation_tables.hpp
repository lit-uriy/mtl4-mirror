// $COPYRIGHT$

#ifndef MTL_MASKED_DILATION_TABLES_INCLUDE
#define MTL_MASKED_DILATION_TABLES_INCLUDE

namespace mtl { namespace dilated {

template <class T, T Mask>
struct masked_dilation_tables
{
    static const unsigned n_bytes= sizeof(T);           // number of bytes of the unmasked value of this type

    typedef masked_dilation_tables     self;
    typedef T                          lookup_type[n_bytes][256];
    typedef T                          mp_type[n_bytes];
    typedef int                        it_type[n_bytes];  // int table type

protected:
    static lookup_type*                my_mask_lut;
    static lookup_type*                my_unmask_lut;
    static mp_type*                    my_mask_piece;
    static it_type*                    my_mask_size;
    static it_type*                    my_mask_shift_table;
    static it_type*                    my_unmask_shift_table;
    static int                         n_valid_table;
   
public:
    lookup_type& mask_lut()
    {
	if (my_mask_lut == 0) compute_tables();
	return *my_mask_lut;
    }

    lookup_type& unmask_lut()
    {
	if (my_unmask_lut == 0) compute_tables();
	return *my_unmask_lut;
    }

    mp_type& mask_piece()
    {
	if (my_mask_piece == 0) compute_tables();
	return *my_mask_piece;
    }

    it_type& mask_size()
    {
	if (my_mask_size == 0) compute_tables();
	return *my_mask_size;
    }

    it_type& mask_shift_table()
    {
	if (my_mask_shift_table == 0) compute_tables();
	return *my_mask_shift_table;
    }

    it_type& unmask_shift_table()
    {
	if (my_unmask_shift_table == 0) compute_tables();
	return *my_unmask_shift_table;
    }

private:

    // get mask of the style 0xfff...
    static T get_f_mask(int n_bits) 
    {
	return (1 << n_bits) - 1;
    }

    T inc(T i, T mask) 
    {
	return ((i - mask) & mask);
    }

    void compute_tables() 
    {
	// std::cout << "computing tables! " << std::endl;
	init();

	// compute the mask table
	for (int j = 0; j < n_valid_table; ++j) {
	    T f_mask = get_f_mask(mask_size()[j]);
	    T ii;
	    int i;
	    for (i = 0, ii = 0; i < 256; ++i, ii = inc(ii, mask_piece()[j])) {
		mask_lut()[j][i] =  (ii & f_mask) << mask_shift_table()[j]; // need to shift 
	    }
	}

	// compute the unmask table
	T f_mask = get_f_mask(8);
	for (int j = 0; j < sizeof(T); ++j) {
	    
	    T t_mask = (Mask >> (8*j)) & f_mask;
	    T ii;
	    int i;
	    for(i = 0, ii = 0; ii < t_mask; ii = inc(ii, t_mask), ++i) {
		unmask_lut()[j][ii] =  i << unmask_shift_table()[j];
	    }
	    // set the value for the last one
	    unmask_lut()[j][t_mask] =  i << unmask_shift_table()[j];       
	}
    }


    void allocate()
    {
	my_mask_lut=   new lookup_type[1];
	my_unmask_lut= new lookup_type[1];
	my_mask_piece= new mp_type[1];
	my_mask_size=  new it_type[1];
	my_mask_shift_table=  new it_type[1];
	my_unmask_shift_table=  new it_type[1];
    }


    // initialize needed parameters
    void init() 
    {
	allocate();

	// calculate the number of valid table
	int n = count_n_ones(Mask);
	if (n <= 8) 
	    n_valid_table = 1;
	else n_valid_table = (n%8 == 0 ? n/8 : n/8 + 1);           
	
	// set mask pieces 
	set_mask();    
    }


    // return the number of 1's in the mask
    int count_n_ones(T t) 
    {
	int n_ones = 0;
	while(t) {
	    if((t & 0x01) == 1) ++n_ones;
	    t = t >>1;
	}
	return n_ones;
    };


    // return the number of valid bits in the mask
    int count_bits(T t) 
    {
	int bits = 0;
	while(t) {
	    ++bits;
	    t = t >>1;
	}
	return bits;
    };


    // set mask pieces
    void set_mask() 
    {
	// set the unmask shift table
	unmask_shift_table()[0] = 0;
	T t_mask = Mask;
	T tmp;
	int count;
	for (int i = 1; i < n_bytes; ++i) {
	    tmp = t_mask & get_f_mask(8);
	    count = count_n_ones(tmp);
	    unmask_shift_table()[i] = count + unmask_shift_table()[i - 1];
	    t_mask = t_mask >> (8*i);
	}

	mask_shift_table()[0] = 0;  // don't need shift for the first table
	// if there is only 8 or less 1's in the mask,
	// only one table is needed
	if (n_valid_table == 1) {
	    mask_piece()[0] = Mask; 
	    mask_size()[0] = count_bits(Mask);
	    return;
	}

	t_mask = Mask;
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
	    mask_piece()[i] = get_f_mask(n_bits) & tmp;
	    
	    // set the mask size table
	    mask_size()[i] = n_bits;
	    
	    // set shift table
	    mask_shift_table()[i + 1] = n_bits + mask_shift_table()[i];
	}

	// set the last piece of mask, which may contain less than 8 1's
	// set the number of bits of the last mask
	mask_piece()[n_valid_table - 1 ] = t_mask;    
	mask_size()[n_valid_table - 1] = count_bits(t_mask);
    };

public:
    
    void check()
    {
	if (n_valid_table == 0)
	    compute_tables();
    }	

    // convert to masked integer
    T to_masked(T x) 
    {
	check();
	T result = 0;
	for (int i = 0; i < n_valid_table; ++i)
	    result += mask_lut()[i][0xff & (x >> (8*i)) ];
	return result;
    }


    // convert to unmasked integer
    T to_unmasked(T x) 
    {
	check();
	T result = 0;
	x &= Mask;
	for (int i = 0; i < n_bytes; ++i)
	    result += unmask_lut()[i][0xff & (x >> (8*i)) ];
	return result;
    }

protected:
    // T value;               // value of the integer
};

template <class T, T Mask>
typename masked_dilation_tables<T, Mask>::lookup_type* masked_dilation_tables<T, Mask>::my_mask_lut= 0;

template <class T, T Mask>
typename masked_dilation_tables<T, Mask>::lookup_type* masked_dilation_tables<T, Mask>::my_unmask_lut= 0;

template <class T, T Mask>
typename masked_dilation_tables<T, Mask>::mp_type* masked_dilation_tables<T, Mask>::my_mask_piece= 0;

template <class T, T Mask>
typename masked_dilation_tables<T, Mask>::it_type* masked_dilation_tables<T, Mask>::my_mask_size= 0;

template <class T, T Mask>
typename masked_dilation_tables<T, Mask>::it_type* masked_dilation_tables<T, Mask>::my_mask_shift_table= 0;

template <class T, T Mask>
typename masked_dilation_tables<T, Mask>::it_type* masked_dilation_tables<T, Mask>::my_unmask_shift_table= 0;

template <class T, T Mask>
int masked_dilation_tables<T, Mask>::n_valid_table= 0;


} // namespace mtl::dilated

  //using dilated::dilated_int;

} // namespace mtl

#endif // MTL_MASKED_DILATION_TABLES_INCLUDE
