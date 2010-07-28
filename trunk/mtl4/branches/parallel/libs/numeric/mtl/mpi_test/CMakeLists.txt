project (boost_mtl4_mpi_test)

# DERIVED variables
# none in this CMakeLists.txt

# collect the source files
file (GLOB SOURCES mpi*test.cpp)
# collect the include files
file (GLOB INCLUDES ${MTL_BOOST_ROOT}/boost/numeric/mtl/*.hpp) # don't think we need this 

#define the include directories this library depends on
include_directories("${MY_BOOST_ROOT}" "${MTL_BOOST_ROOT}")
message(STATUS "BOOST_ROOT is ${MY_BOOST_ROOT}")

set(MPI_INCLUDE_PATH ${MPI_INCLUDE_PATH} "${MY_BOOST_ROOT}" "${MTL_BOOST_ROOT}")

# Only compile this source, don't link it
# get_filename_component (test mpi_2_mpi_log_second.cpp NAME_WE)		# filename without extension
# add_custom_command(OUTPUT ${test} DEPENDS ${source} COMMAND ${MPI_COMPILER} ${source} -o  ${test}  )
# add_custom_target(${test}_helper ALL DEPENDS ${test} )

set (INC_FLAGS "-I${MTL_BOOST_ROOT}" "-I${MY_BOOST_ROOT}") # to do: portable
set (LIB_FLAGS ${CMAKE_LIBRARY_PATH_FLAG} ${MY_BOOST_ROOT}/lib ${CMAKE_LINK_LIBRARY_FLAG} boost_serialization ${CMAKE_LINK_LIBRARY_FLAG} boost_mpi) # todo find path automatically
set (MACRO_FLAGS "-DMTL_HAS_MPI" "-DBOOST_MPI_HOMOGENEOUS" "-Wno-array-bounds") # todo: portable

# cycle through the sources
# The directory assumes that each cpp file is a separate test
# so, create a executable target and an associated test target
foreach (source ${SOURCES})
    get_filename_component (test ${source} NAME_WE)		# filename without extension
    if ("${test}" STREQUAL "mpi_2_mpi_log_test")
      message ("Ignore mpi_2_mpi_log_test for the moment")
      # add_custom_command(OUTPUT mpi_2_mpi_log_second.o DEPENDS mpi_2_mpi_log_second.cpp COMMAND ${MPI_COMPILER} ${MACRO_FLAGS} ${INC_FLAGS} -c mpi_2_mpi_log_second.cpp -o mpi_2_mpi_log_second.o)
      # add_custom_target(mpi_2_mpi_log_second_helper mpi_2_mpi_log_second.o)

      # add_custom_command(OUTPUT ${test} DEPENDS ${source} COMMAND ${MPI_COMPILER} ${MACRO_FLAGS} ${INC_FLAGS} ${source} mpi_2_mpi_log_second.o -o ${test} ${LIB_FLAGS})
      # add_custom_target(${test}_helper ALL DEPENDS ${test} )
      # add_dependencies(${test}_helper mpi_2_mpi_log_second.o)
    else()
      add_custom_command(OUTPUT ${test} DEPENDS ${source} COMMAND ${MPI_COMPILER} ${MACRO_FLAGS} ${INC_FLAGS} ${source} -o ${test} ${LIB_FLAGS})
      add_custom_target(${test}_helper ALL DEPENDS ${test} )
    endif () 

    set(test_name "${test}")
    # message(STATUS "${test_name}")
    string(REGEX REPLACE "mpi_([0-9]*)_.*" "\\1" nprocs "${test_name}")
    add_test (${test} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${nprocs} ${EXECUTABLE_OUTPUT_PATH}/${test})
endforeach (source)


