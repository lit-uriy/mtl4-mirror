project (boost_mtl4)

cmake_minimum_required(VERSION 2.6)

# DERIVED VARIABLES
# no derived variables, this makefile is the toplevel makefile for boost mtl4

# The BOOST_INCLUDE_DIR needs to point to the release BOOST tree
# so that we can pick up boost/test/minimal.hpp
# set (BOOST_ROOT "please-set-boost-include-dir" CACHE STRING "BOOST include directory")

set (MY_BOOST_ROOT "$ENV{BOOST_ROOT}" CACHE PATH "BOOST include directory") # CMake gets confused when same name in env and here
find_path(Boost_INCLUDE_DIR boost/version.hpp PATHS "${MY_BOOST_ROOT}")
if(NOT Boost_INCLUDE_DIR)
  message(FATAL_ERROR "boost wasn't found, please set BOOST_ROOT")
endif(NOT Boost_INCLUDE_DIR)

find_package(Boost)

# set the local BOOST MTL4 include directory
# to pick up the MTL4 include files
set (MTL_BOOST_ROOT ${CMAKE_CURRENT_SOURCE_DIR} CACHE STRING "BOOST MTL4 root directory")

# include_directories(${Boost_INCLUDE_DIR})
include_directories(${MY_BOOST_ROOT} ${MTL_BOOST_ROOT})

find_package(MPI)

## Extra definitions for Boost MPI (maybe write module FindBoostMPI
set (PMTL_INC_FLAGS "-I${MTL_BOOST_ROOT}" "-I${MY_BOOST_ROOT}") # to do: portable
set (PMTL_LIB_FLAGS ${CMAKE_LIBRARY_PATH_FLAG} ${MY_BOOST_ROOT}/lib ${CMAKE_LINK_LIBRARY_FLAG} boost_serialization ${CMAKE_LINK_LIBRARY_FLAG} boost_mpi) # todo find path automatically
set (PMTL_MACRO_FLAGS "-DMTL_HAS_MPI" "-DBOOST_MPI_HOMOGENEOUS" "-Wno-array-bounds") # todo: portable


#################################################################################
######################## DevStudio funkyness   ################################
#################################################################################
if (MSVC80 OR MSVC90)
	# to quiet DevStudio 2005 secure code warnings
    add_definitions(/D_SCL_SECURE_NO_WARNINGS)
    add_definitions(/D_CRT_SECURE_NO_WARNINGS)
    add_definitions(/DMTL_ASSERT_FOR_THROW /D_CRT_SECURE_NO_DEPRECATE /DNOMINMAX /D_CONSOLE /D"_HAS_ITERATOR_DEBUGGING=0" /D"_SECURE_SCL=0" )
	# specialty flags such as the C++ language exception model etc. come from the init flags determined by CMAKE
	# you may override them, but I have not found it necessary
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG_INIT} /wd4018 /wd4099 /wd4522 /wd4996 /wd4355 /Z7")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE_INIT} /wd4018 /wd4099 /wd4522 /wd4996 /wd4355 /wd4244 /Z7")
endif (MSVC80 OR MSVC90)

#################################################################################
######################## Macros for sub-directories   ###########################
#################################################################################

macro (compile_all)
    # cycle through the sources
    # The directory assumes that each cpp file is a separate test
    # so, create a executable target and an associated test target
    foreach (source ${ARGV})
        get_filename_component (test ${source} NAME_WE)		# filename without extension
        add_executable (${test} ${source})
        add_test (${test} ${EXECUTABLE_OUTPUT_PATH}/${test})
    endforeach (source)
endmacro (compile_all)

# same with MPI (boost::mpi)
macro (mpi_compile_all)
    foreach (source ${ARGV})
        get_filename_component (test ${source} NAME_WE)		# filename without extension
        add_custom_command(OUTPUT ${test} DEPENDS ${source} COMMAND ${MPI_COMPILER} ${PMTL_MACRO_FLAGS} ${PMTL_INC_FLAGS} ${source} -o ${test} ${PMTL_LIB_FLAGS})
        add_custom_target(${test}_helper ALL DEPENDS ${test} )
        set(test_name "${test}")
        # message(STATUS "${test_name}")
        string(REGEX REPLACE "mpi_([0-9]*)_.*" "\\1" nprocs "${test_name}")
        add_test (${test} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${nprocs} ${EXECUTABLE_OUTPUT_PATH}/${test})
    endforeach (source)
endmacro (mpi_compile_all)


# enable test generation
enable_testing()

# Descent into the test program directory to build the test suite
# MTL4 test programs
add_subdirectory("libs/numeric/mtl/test")
add_subdirectory("libs/numeric/mtl/mpi_test")

# as well as the example programs
add_subdirectory("libs/numeric/mtl/examples")
add_subdirectory("libs/numeric/mtl/mpi_examples")

# ITL test programs
add_subdirectory("libs/numeric/itl/test")
add_subdirectory("libs/numeric/itl/mpi_test")

#################################################################################
## handy for learning CMAKE and debugging possible problems
######################## record all the variables CMAKE defined #################
file (WRITE ${CMAKE_CURRENT_BINARY_DIR}/AllVariables.txt "")
get_cmake_property(VARS VARIABLES)
foreach (var ${VARS})
    file (APPEND ${CMAKE_CURRENT_BINARY_DIR}/AllVariables.txt "${var} \"${${var}}\"\n")
endforeach (var ${VARS})





