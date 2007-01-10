// $COPYRIGHT$

#include <iostream>
#include <vector>
#include <boost/test/minimal.hpp>
#include <boost/timer.hpp>

#include <boost/numeric/mtl/dense2D.hpp>
#include <boost/numeric/mtl/operations/hessian_matrix_utilities.hpp>

using namespace std;
using namespace mtl;


void print_time_and_mflops(double time, double size)
{ 
    // std::cout << "    takes " << time << "s = " << 2.0 * size * size * size / time / 1e6f << "MFlops\n";
    std::cout << size << ", " << time << ", " << 2.0 * size * size * size / time / 1e6f << "\n";
    std::cout.flush();
}


// Matrices are only placeholder to provide the type
template <typename MatrixA, typename MatrixB, typename MatrixC, typename Mult>
double time_measure(MatrixA&, MatrixB&, MatrixC&, Mult mult, unsigned size)
{
    MatrixA a(size, size);
    MatrixB b(size, size);
    MatrixC c(size, size);

    fill_hessian_matrix(a, 1.0);
    fill_hessian_matrix(b, 2.0); 

    // repeat multiplication if it is less than a second (until it is a second)
    int i; boost::timer start1;
    for (i= 0; start1.elapsed() < 1.0; i++)
	mult(a, b, c);
    double elapsed= start1.elapsed() / double(i);
    print_time_and_mflops(elapsed, size);
    return elapsed;
}
 
template <typename MatrixA, typename MatrixB, typename MatrixC, typename Mult>
void time_series(MatrixA& a, MatrixB& b, MatrixC& c, Mult mult, const char* name, unsigned steps, unsigned max_size)
{
    // Maximal time per measurement 20 min
    double max_time= 1200.0;

    std::cout << "\n# " << name << ":\n";
    std::cout << "# Gnu-Format size, time, MFlops\n";
    std::cout.flush();

    for (unsigned i= steps; i <= max_size; i+= steps) {
	double elapsed= time_measure(a, b, c, mult, i);
	if (elapsed > max_time) break;
    }
}


void mult_simple(const dense2D<double>& a, const dense2D<double>& b, dense2D<double>& c)
{
    for (unsigned i= 0; i < c.num_rows(); i++)
	for (unsigned k= 0; k < c.num_cols(); k++) {
	    double tmp= 0.0;
	    for (unsigned j= 0; j < b.num_cols(); j++)
		tmp+= a[i][j] * b[j][k];
	    c[i][k]= tmp;
	}
}

void mult_simple_p(dense2D<double>& a, dense2D<double>& b, dense2D<double>& c)
{
    for (unsigned i= 0; i < c.num_rows(); i++)
	for (unsigned k= 0; k < c.num_cols(); k++) {
	    double tmp= 0.0;
	    double *begin_a= &a[i][0], *end_a= &a[i][a.num_cols()], *begin_b= &b[0][k];
	    int ld= b.num_rows();
	    for (; begin_a != end_a; ++begin_a, begin_b+= ld)
		tmp+= *begin_a * *begin_b;
	    c[i][k]= tmp;
	}
}

typedef dense2D<double, matrix_parameters<col_major> >   cm_type;

void mult_simple_pt(dense2D<double>& a, cm_type& b, dense2D<double>& c)
{
    for (unsigned i= 0; i < c.num_rows(); i++)
	for (unsigned k= 0; k < c.num_cols(); k++) {
	    double tmp= 0.0;
	    double *begin_a= &a[i][0], *end_a= &a[i][a.num_cols()], *begin_b= &b[0][k];
	    for (; begin_a != end_a; ++begin_a, ++begin_b)
		tmp+= *begin_a * *begin_b;
	    c[i][k]= tmp;
	}
}


void mult_simple_ptu(dense2D<double>& a, cm_type& b, dense2D<double>& c)
{
    for (unsigned i= 0; i < c.num_rows(); i++)
	for (unsigned k= 0; k < c.num_cols(); k+=2) {
	    int ld= b.num_rows();
	    double tmp0= 0.0, tmp1= 0.0;

	    double *begin_a= &a[i][0], *end_a= &a[i][a.num_cols()];
	    double *begin_b= &b[0][k];
	    for (; begin_a != end_a; ++begin_a, ++begin_b) {
		tmp0+= *begin_a * *begin_b;
		tmp1+= *begin_a * *(begin_b+ld);
	    }
	    c[i][k]= tmp0; c[i][k+1]= tmp1;
	}
}

void mult_simple_ptu4(dense2D<double>& a, cm_type& b, dense2D<double>& c)
{
    for (unsigned i= 0; i < c.num_rows(); i++)
	for (unsigned k= 0; k < c.num_cols(); k+=2) {
	    int ld1= b.num_rows(), ld2= 2*ld1, ld3=3*ld1;
	    double tmp0= 0.0, tmp1= 0.0, tmp2= 0.0, tmp3= 0.0;

	    double *begin_a= &a[i][0], *end_a= &a[i][a.num_cols()];
	    double *begin_b= &b[0][k];
	    for (; begin_a != end_a; ++begin_a, ++begin_b) {
		tmp0+= *begin_a * *begin_b;
		tmp1+= *begin_a * *(begin_b+ld1);
		tmp2+= *begin_a * *(begin_b+ld2);
		tmp3+= *begin_a * *(begin_b+ld3);
	    }
	    c[i][k]= tmp0; c[i][k+1]= tmp1;
	    c[i][k+2]= tmp2; c[i][k+3]= tmp3;
	}
}

// C must be square!
void mult_simple_ptu22(dense2D<double>& a, cm_type& b, dense2D<double>& c)
{
    for (unsigned i= 0; i < c.num_rows(); i+=2)
	for (unsigned k= 0; k < c.num_cols(); k+=2) {
	    int ld= b.num_rows();
	    double tmp00= 0.0, tmp01= 0.0, tmp10= 0.0, tmp11= 0.0;

	    double *begin_a= &a[i][0], *end_a= &a[i][a.num_cols()];
	    double *begin_b= &b[0][k];
	    for (; begin_a != end_a; ++begin_a, ++begin_b) {
		tmp00+= *begin_a * *begin_b;
		tmp01+= *begin_a * *(begin_b+ld);
		tmp10+= *begin_a * *begin_b;
		tmp11+= *(begin_a+ld) * *(begin_b+ld);
	    }
	    c[i][k]= tmp00; c[i][k+1]= tmp01;
	    c[i+1][k]= tmp10; c[i+1][k+1]= tmp11;
	}
}


int test_main(int argc, char* argv[])
{
    unsigned steps= 32, max_size= 128, size= 32; 
    if (argc > 2) {
	steps= atoi(argv[1]); max_size= atoi(argv[2]);
    }
    
    mtl::dense2D<double>               da(size, size), db(size, size), dc(size, size);
    cm_type                            dat(size, size), dbt(size, size), dct(size, size);
    fill_hessian_matrix(da, 1.0);
    fill_hessian_matrix(db, 2.0); 
    fill_hessian_matrix(dbt, 2.0); 

    time_series(da, dbt, dc, mult_simple_ptu22, "Simple mult (pointers trans unrolled 2x2)", steps, max_size);
    time_series(da, dbt, dc, mult_simple_ptu4, "Simple mult (pointers trans unrolled 4)", steps, max_size);
    time_series(da, dbt, dc, mult_simple_ptu, "Simple mult (pointers trans unrolled 2)", steps, max_size);
    time_series(da, dbt, dc, mult_simple_pt, "Simple mult (pointers transposed)", steps, max_size);
    time_series(da, db, dc, mult_simple_p, "Simple mult (pointers)", steps, max_size);
    time_series(da, db, dc, mult_simple, "Simple mult", steps, max_size);


    return 0;
}
