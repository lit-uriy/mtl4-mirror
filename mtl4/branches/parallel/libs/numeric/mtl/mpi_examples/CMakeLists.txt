project (boost_mtl4_mpi_examples)

# collect the source files and compile them
file (GLOB SOURCES mpi*.cpp)
mpi_compile_all(${SOURCES})

