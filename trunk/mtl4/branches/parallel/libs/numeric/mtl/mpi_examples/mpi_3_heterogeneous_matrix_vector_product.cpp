// Filename: mpi_3_heterogeneous_matrix_vector_product.cpp

#include <iostream>
#include <boost/mpi.hpp>
#include <boost/numeric/mtl/mtl.hpp>

int main(int argc, char* argv[]) 
{
    boost::mpi::environment      env(argc, argv);
    boost::mpi::communicator     world;

    std::size_t                  ra[]= {0, 4, 6, 7}, ca[]= {0, 5, 8, 8};
    mtl::par::block_distribution row_dist= ra,       col_dist= ca;

    typedef mtl::matrix::distributed<mtl::compressed2D<float> >  matrix_type;
    matrix_type A(7, 8, row_dist, col_dist);

    A= 4.0;

    mtl::vector::distributed<mtl::dense_vector<double> > u(7, row_dist), 
	                                                 v(8, col_dist, 3.0);
    
    u= A * v;

    mtl::par::sout << "The vector v is          " << v 
		   << "\nand the product A * v is " << u << "\n";
    return 0;
}
