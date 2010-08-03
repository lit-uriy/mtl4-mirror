// Filename: mpi_3_single_ostream.cpp

#include <iostream>
#include <boost/mpi.hpp>
#include <boost/numeric/mtl/mtl.hpp>

int main(int argc, char* argv[]) 
{
    boost::mpi::environment env(argc, argv);

    for (int i= 0; i < 7; i++) {
	// compute something
	mtl::par::sout << "Iteration " << i << ": x = \n";
    }

    return 0;
}
