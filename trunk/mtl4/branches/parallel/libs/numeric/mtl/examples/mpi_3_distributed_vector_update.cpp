// Filename: mpi_3_distributed_vector_update.cpp

#include <iostream>
#include <boost/mpi.hpp>
#include <boost/numeric/mtl/mtl.hpp>

int main(int argc, char* argv[]) 
{
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    typedef mtl::vector::distributed<mtl::dense_vector<float> >  vector_type;
    vector_type  v(8, 0.0f);
    
    {
	mtl::vector::inserter<vector_type, mtl::update_plus<float> > ins(v);
	for (unsigned i= 0; i < size(v); ++i)
	    if (i % world.size() == world.rank())
		ins[i] << float(world.rank());
	    else
		ins[i] << 0.1;
    }

    std::cout << "I am proc. " << world.rank() << " and my local part of v is " << local(v) << '\n';

    return 0;
}
