project (boost_mtl4_mpi_test)

# collect the source files and compile them
file (GLOB SOURCES mpi*test.cpp)
mpi_compile_all(${SOURCES})

message ("Ignore mpi_2_mpi_log_first for the moment")
# add_custom_command(OUTPUT mpi_2_mpi_log_second.o DEPENDS mpi_2_mpi_log_second.cpp COMMAND ${MPI_COMPILER} ${PMTL_MACRO_FLAGS} ${PMTL_INC_FLAGS} -c mpi_2_mpi_log_second.cpp -o mpi_2_mpi_log_second.o)
# add_custom_target(mpi_2_mpi_log_second_helper mpi_2_mpi_log_second.o)
# add_custom_command(OUTPUT ${test} DEPENDS ${source} COMMAND ${MPI_COMPILER} ${PMTL_MACRO_FLAGS} ${PMTL_INC_FLAGS} ${source} mpi_2_mpi_log_second.o -o ${test} ${PMTL_LIB_FLAGS})
# add_custom_target(${test}_helper ALL DEPENDS ${test} )
# add_dependencies(${test}_helper mpi_2_mpi_log_second.o)
