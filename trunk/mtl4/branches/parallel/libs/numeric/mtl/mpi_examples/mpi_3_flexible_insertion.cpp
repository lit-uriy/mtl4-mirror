// Filename: mpi_3_flexible_insertion.cpp

#include <iostream>
#include <boost/mpi.hpp>
#include <boost/numeric/mtl/mtl.hpp>

int main(int argc, char* argv[]) 
{
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    typedef mtl::matrix::distributed<mtl::compressed2D<float>,
	mtl::par::block_cyclic_distribution,
	mtl::par::cyclic_distribution>                         matrix_type;

    mtl::par::block_cyclic_distribution  row_dist(2); 
    mtl::par::cyclic_distribution        col_dist;
    matrix_type A(7, 7, row_dist, col_dist);

    {
	mtl::matrix::inserter<matrix_type> ins(A);
	if (world.rank() == 0) {
	    ins[0][0] << 1.0;
	    ins[1][3] << 3.0;
	    ins[4][1] << 2.0; // remote
	    ins[6][5] << 4.0; // remote
	} else if (world.rank() == 2) {
	    ins[2][6] << 5.0; // remote
	    ins[3][2] << 6.0; // remote
	    ins[5][4] << 8.0; // remote
	}
    }

    return 0;
}
