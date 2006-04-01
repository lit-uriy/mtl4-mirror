// $COPYRIGHT$

#include <iostream>
// #include <boost/test/minimal.hpp>

#include <boost/numeric/mtl/compressed2D.hpp>
#include <boost/numeric/mtl/matrix_parameters.hpp>


#include <boost/numeric/mtl/operations/update.hpp>

using namespace mtl;
using namespace std;

template <typename Generator>
void generator_overhead(int s, Generator gen)
{
}

template <typename Generator>
void insert_petsc(int s, Generator gen, double overhead)
{
}

template <typename Generator>
void insert_mtl2(int s, Generator gen, double overhead)
{
}

template <typename Generator>
void insert_mtl4(int s, Generator gen, double overhead)
{
    typedef typename Generator::value_type                                             value_type;
    typedef matrix_parameters<row_major, mtl::index::c_index, non_fixed::dimensions>   parameters;
    typedef compressed2D<value_type, parameters>                                       matrix_type;
    matrix_type   matrix(non_fixed::dimensions(s, s)); 
  
    // time t;
    compressed2D_inserter<value_type, parameters>  inserter(matrix);
    while (! gen.finished() ) {
	int      r, c;
	double   v;
	gen(r, c, v);
	inserter(r, c) << v;
	// cout << "A[" << r << ", " << c << "] = " << v << '\n';
    }
    // t.elapsed() - overhead ==> gnuplot or whatever
}



template <typename Size, typename Value>
struct poisson_generator
{
    typedef Value  value_type;

    explicit poisson_generator(int s) : s(s), count(0), row(0), offset(2) {     // s must be 2^k 100
	d1= 10, d2= s/d1;
	for (; d2 > d1; d1<<= 1) d2= s / d1;
	// d1= d2= 3;                                     // only to test a 9x9 matrix
	nnz= 5 * s - 2 * d1 - 2 * d2;
    }
    
    bool finished() { 
	return count >= nnz; 
    }

    bool valid_offset() {
	switch (offset) {
	    case 0: return row >= d2; // northern nb
	    case 1: return row % d2;  // western nb
	    case 2: return true;
	    case 3: return (row + 1) % d2; // eastern nb
	    case 4: return row + d2 < s;   // southern nb
	}
	assert(true); // shouldn't be reached
	return false;
    }

    void next_offset() {
	offset++;
	if (offset == 5) offset= 0, row++;
    }

    void next_nnz() {
	do 
	    {next_offset(); } 
	while (! valid_offset());
	count++;
    }

    void operator() (Size& r, Size& c, Value& v) {
	r= row;
	switch (offset) {
	    case 0: c= r - d2; v= -1; break;
	    case 1: c= r - 1; v= -1; break;
	    case 2: c= r; v= -4; break;
	    case 3: c= r + 1; v= -1; break;
	    case 4: c= r + d2; v= -1; 
	}
	next_nnz();
    }	    

    int s, d1, d2, count, 
	nnz,                   // number of non-zeros in matrix
	row, offset;           // current row and which entry in the row
};



template <typename Generator>
void run(int max_size)
{
    for (int s= 100; s < max_size; s*= 2) {
	Generator gen0(s);
	double overhead= generator_overhead(s, gen0);

	Generator gen1(s);
	insert_petsc(s, gen1, overhead);

	Generator gen2(s);
	insert_mtl2(s, gen2, overhead);
	
	Generator gen3(s);
	insert_mtl4(s, gen3, overhead);
    }
}

void check_dims()
{
    for (int s= 100;  s < 1000000; s*= 2) {
	int d1= 10, d2= s/d1;
	for (; d2 > d1; d1<<= 1) 
	    d2= s / d1;
	std::cout << d1 << " * " << d2 << " = " << s << '\n';
    }
}

int main(int argc, char* argv[])
{
    // check_dims();
    // poisson_generator<int, double> poisson_9(9);
    // insert_mtl4(9, poisson_9, 0.0);

    // run<poisson_generator<int, double> > (atoi(argv[1]));

    return 0;
}
 
