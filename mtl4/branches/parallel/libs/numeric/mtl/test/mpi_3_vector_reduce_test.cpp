#include <boost/mpi.hpp>
#include <iostream>
#include <cstdlib>
#include <boost/numeric/mtl/mtl.hpp>
#include <functional>

namespace mpi = boost::mpi;

template <typename Vector>
struct vector_plus
{
    Vector operator()(const Vector& x, const Vector& y)
    {
	return Vector(x + y);  // because implicit conversion from expression template to vector is disabled
    }
};


int main(int argc, char* argv[])
{
    mpi::environment env(argc, argv);
    mpi::communicator world;

    std::srand(time(0) + world.rank());
    int my_number = std::rand();

    typedef mtl::dense_vector<double>                     vector_type;
    typedef vector_plus<vector_type> plus;

    vector_type vrand(3), sum(3);
    random(vrand);

    if (world.rank() == 0) {
	reduce(world, vrand, sum, plus(), 0);
	std::cout << "The sum of all vectors is " << sum << std::endl;
    } else {
	reduce(world, vrand, plus(), 0);
    }  
    return 0;
}

 
