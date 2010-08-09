// Filename: mpi_3_heterogeneous_matrix_vector_product.cpp

#include <iostream>
#include <boost/mpi.hpp>
#include <boost/numeric/mtl/mtl.hpp>

int main(int argc, char* argv[]) 
{
    boost::mpi::environment      env(argc, argv);
    boost::mpi::communicator     world;

    std::size_t                  ra[]= {0, 4, 6, 7}, ca[]= {0, 5, 7, 7};
    mtl::par::block_distribution row_dist= ra,       col_dist= ca;

    typedef mtl::matrix::distributed<mtl::compressed2D<float> >  matrix_type;
    matrix_type A(7, 7, row_dist, col_dist);

    {
	mtl::matrix::inserter<matrix_type> ins(A);
	if (world.rank() == 0) {
	    ins[0][0] << 1.0;
	    ins[1][3] << 3.0;
	    ins[4][1] << 2.0; 
	    ins[6][5] << 4.0; 
	    ins[2][6] << 5.0; 
	    ins[3][2] << 6.0; 
	    ins[5][4] << 8.0; 
	}
    }

    mtl::par::sout << "The matrix A is\n" << A << "\n";
    return 0;
}
