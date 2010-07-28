project (boost_mtl4_itl)

# DERIVED variables
# none in this CMakeLists.txt
# This makefile is meant to be the toplevel for the examples

# The BOOST_INCLUDE_DIR needs to point to the release BOOST tree
# so that we can pick up boost/test/minimal.hpp
set (BOOST_ROOT "please-set-boost-include-dir" CACHE STRING "BOOST include directory")

# set the local BOOST MTL4 include directory
# to pick up the MTL4 include files
set (MTL_BOOST_ROOT "please-set-mtl4-root-dir" CACHE STRING "BOOST MTL4 root directory")


# collect the source files
file (GLOB SOURCES *.cpp)

# enable testing
enable_testing()

# cycle through the sources
# The directory assumes that each cpp file is a separate test
# so, create a executable target and an associated test target
foreach (source ${SOURCES})
    get_filename_component (test ${source} NAME_WE)		# NAME_WE == fileName Without Extension
    add_executable (${test} ${source})
    add_test (${test} ${EXECUTABLE_OUTPUT_PATH}/${test})
endforeach (source)


