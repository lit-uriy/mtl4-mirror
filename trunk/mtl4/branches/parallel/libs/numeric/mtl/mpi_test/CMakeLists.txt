project (boost_mtl4_mpi_test)

# DERIVED variables
# none in this CMakeLists.txt

# collect the source files
file (GLOB SOURCES mpi*test.cpp)
# collect the include files
file (GLOB INCLUDES ${MTL_BOOST_ROOT}/boost/numeric/mtl/*.hpp) # don't think we need this 

message ("Ignore mpi_2_mpi_log_first for the moment")
# add_custom_command(OUTPUT mpi_2_mpi_log_second.o DEPENDS mpi_2_mpi_log_second.cpp COMMAND ${MPI_COMPILER} ${PMTL_MACRO_FLAGS} ${PMTL_INC_FLAGS} -c mpi_2_mpi_log_second.cpp -o mpi_2_mpi_log_second.o)
# add_custom_target(mpi_2_mpi_log_second_helper mpi_2_mpi_log_second.o)
# add_custom_command(OUTPUT ${test} DEPENDS ${source} COMMAND ${MPI_COMPILER} ${PMTL_MACRO_FLAGS} ${PMTL_INC_FLAGS} ${source} mpi_2_mpi_log_second.o -o ${test} ${PMTL_LIB_FLAGS})
# add_custom_target(${test}_helper ALL DEPENDS ${test} )
# add_dependencies(${test}_helper mpi_2_mpi_log_second.o)


# cycle through the sources
# The directory assumes that each file like mpi*test.cpp is a separate test
# so, create a executable target and an associated test target
foreach (source ${SOURCES})
    get_filename_component (test ${source} NAME_WE)		# filename without extension
    add_custom_command(OUTPUT ${test} DEPENDS ${source} COMMAND ${MPI_COMPILER} ${PMTL_MACRO_FLAGS} ${PMTL_INC_FLAGS} ${source} -o ${test} ${PMTL_LIB_FLAGS})
    add_custom_target(${test}_helper ALL DEPENDS ${test} )
    set(test_name "${test}")
    # message(STATUS "${test_name}")
    string(REGEX REPLACE "mpi_([0-9]*)_.*" "\\1" nprocs "${test_name}")
    add_test (${test} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${nprocs} ${EXECUTABLE_OUTPUT_PATH}/${test})
endforeach (source)


