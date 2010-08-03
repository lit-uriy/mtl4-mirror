// Filename: mpi_3_vector_temporary.cpp

#include <iostream>
#include <boost/mpi.hpp>
#include <boost/numeric/mtl/mtl.hpp>

int main(int argc, char* argv[]) 
{
    boost::mpi::environment      env(argc, argv);
    std::size_t                  array[]= {0, 2, 7, 10};
    mtl::par::block_distribution dist= array;

    typedef mtl::vector::distributed<mtl::dense_vector<float> >  vector_type;
    vector_type  u(8, 4.0), v(10, dist, 5.0f);
    
    vector_type  w(size(u)),     // Same size and distribution like u
                 x(size(v)),     // Same size as v but different distribution
	         y(v),           // Same size and distribution
	         z(resource(v)); // Likewise but without copying entries

    mtl::par::sout << "w is " << u << "\nx is " << x 
		   << "\ny is " << y << "\nz is " << z << '\n';

    return 0;
}
