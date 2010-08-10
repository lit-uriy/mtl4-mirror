// Filename: mpi_3_heterogeneous_matrix_vector_product.cpp

#include <iostream>
#include <complex>
#include <boost/mpi.hpp>
#include <boost/serialization/complex.hpp>
#include <boost/numeric/mtl/mtl.hpp>

int main(int argc, char* argv[]) 
{
    boost::mpi::environment      env(argc, argv);
    boost::mpi::communicator     world;

    // std::size_t                  ra[]= {0, 4, 6, 7}, ca[]= {0, 5, 8, 8};
    std::size_t                  ra[]= {0, 7, 7, 7}, ca[]= {0, 9, 9, 9};
    mtl::par::block_distribution row_dist= ra,       col_dist= ca;

    typedef mtl::matrix::distributed<mtl::compressed2D<float> >  matrix_type;
    matrix_type A(7, 8, row_dist, col_dist);
    mtl::par::sout << "row_distribution((A)) is " << row_distribution((A)) << "\n";
    mtl::par::sout << "col_distribution((A)) is " << col_distribution((A)) << "\n";

    A= 4.0;

    mtl::vector::distributed<mtl::dense_vector<double> > u(7, row_dist, 3.0), 
	                                                 v(8, col_dist);
    
    v= trans(A) * u;

    mtl::par::sout << "The vector u is                 " << u 
		   << "\nand the product trans(A) * u is " << v << "\n";

    mtl::par::sout << "row_dist is " << row_dist << "\n";
    typedef std::complex<float>   cmplx;
    mtl::matrix::distributed<mtl::compressed2D<cmplx> >   B(7, 8, row_dist, col_dist);
    mtl::par::sout << "row_distribution((B)) is " << row_distribution((B)) << "\n";
    mtl::par::sout << "col_distribution((B)) is " << col_distribution((B)) << "\n";
    mtl::par::sout << "row_distribution(hermitian(B)) is " << row_distribution(hermitian(B)) << "\n";
    B= cmplx(1.0f, 2.0f) * A;

    mtl::vector::distributed<mtl::dense_vector<cmplx> > w(8, col_dist);

    mtl::par::sout << "row_distribution((B)) is " << row_distribution((B)) << "\n";
    mtl::par::sout << "col_distribution((B)) is " << col_distribution((B)) << "\n";
    mtl::par::sout << "row_distribution(hermitian(B)) is " << row_distribution(hermitian(B)) << "\n";

    // w= hermitian(B) * u;
    //mtl::par::sout << "and the product hermitian(B) * u is " << w << "\n";

    return 0;
}
