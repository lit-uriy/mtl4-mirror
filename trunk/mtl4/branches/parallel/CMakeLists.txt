project (boost_mtl4)

cmake_minimum_required(VERSION 2.6)

# DERIVED VARIABLES
# no derived variables, this makefile is the toplevel makefile for boost mtl4

# The BOOST_INCLUDE_DIR needs to point to the release BOOST tree
# so that we can pick up boost/test/minimal.hpp
# set (BOOST_ROOT "please-set-boost-include-dir" CACHE STRING "BOOST include directory")

# set (MY_BOOST_ROOT "$ENV{BOOST_ROOT}" CACHE PATH "BOOST include directory") # CMake gets confused when same name in env and here



# look for boost/version in
# 1. environment variable BOOST_ROOT defined by user
# 2. environment variable BOOST_INC defined on some data centers
# 3. Boost_INCLUDE_DIR found by package
find_package(Boost)
find_path(MY_BOOST_INCLUDE boost/version.hpp "$ENV{BOOST_ROOT}" "$ENV{BOOST_INC}" "${Boost_INCLUDE_DIR}" NO_DEFAULT_PATH)

if(NOT MY_BOOST_INCLUDE)
  message(FATAL_ERROR "boost wasn't found, please set BOOST_ROOT or BOOST_INC")
endif()

message(STATUS "Boost_INCLUDE_DIR is ${Boost_INCLUDE_DIR}")
message(STATUS "MY_BOOST_INCLUDE is ${MY_BOOST_INCLUDE}")

# set the local BOOST MTL4 include directory to pick up the MTL4 include files
set (MTL_BOOST_ROOT ${CMAKE_CURRENT_SOURCE_DIR} CACHE STRING "BOOST MTL4 root directory")

include_directories(${MY_BOOST_INCLUDE} ${MTL_BOOST_ROOT})

find_package(MPI)

## Extra definitions for Boost MPI (maybe write module FindBoostMPI
set (PMTL_INC_FLAGS "-I${MTL_BOOST_ROOT}" "-I${MY_BOOST_INCLUDE}") # to do: portable

set (PMTL_MACRO_FLAGS "-DMTL_HAS_MPI" "-DBOOST_MPI_HOMOGENEOUS")
if (false) # for MPI compilers basing on gcc 4.3 or higher
  set (PMTL_MACRO_FLAGS ${PMTL_MACRO_FLAGS}  "-Wno-array-bounds")
endif()

find_library(BOOST_MPI_LIB_NAME boost_mpi PATHS ${MY_BOOST_ROOT}/lib $ENV{BOOST_LIB} )
get_filename_component(BOOST_MPI_DIR ${BOOST_MPI_LIB_NAME} PATH)
find_library(BOOST_SERIALIZATION_LIB_NAME boost_serialization ${BOOST_MPI_DIR})
if(NOT BOOST_MPI_DIR OR NOT BOOST_SERIALIZATION_LIB_NAME)
  message(FATAL_ERROR "boost::mpi and/or boost::serialization libraries wasn't found, please set BOOST_ROOT or BOOST_LIB")
endif()

# find_library(BOOST_LIB_FIND_LIBS NAMES boost_mpi boost_serialization PATHS ${MY_BOOST_ROOT}/lib $ENV{BOOST_LIB} )
# get_filename_component(BOOST_LIB_DIR ${BOOST_LIB_FIND_LIBS} PATH)
# message(STATUS "BOOST_LIB_DIR is ${BOOST_LIB_DIR}")
# if(NOT BOOST_LIB_DIR)
#   message(FATAL_ERROR "boost::mpi and/or boost::serialization libraries wasn't found, please set BOOST_ROOT or BOOST_LIB")
# endif()


set (PMTL_LIB_FLAGS ${CMAKE_LIBRARY_PATH_FLAG} ${BOOST_MPI_DIR} ${CMAKE_LINK_LIBRARY_FLAG} boost_mpi ${CMAKE_LINK_LIBRARY_FLAG} boost_serialization) # todo find path automatically



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





