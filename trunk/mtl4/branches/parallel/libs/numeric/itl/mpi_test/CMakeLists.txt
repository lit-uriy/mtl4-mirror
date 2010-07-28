project (boost_mtl4_itl_mpi)

# collect the source files
file (GLOB SOURCES *.cpp)
mpi_compile_all(${SOURCES})
