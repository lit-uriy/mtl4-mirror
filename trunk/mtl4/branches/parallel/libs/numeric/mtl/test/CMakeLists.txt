project (boost_mtl4_tests)

# DERIVED variables
# none in this CMakeLists.txt

# collect the source files
file (GLOB SOURCES *.cpp)

# cycle through the sources
# The directory assumes that each cpp file is a separate test
# so, create a executable target and an associated test target
foreach (source ${SOURCES})
    get_filename_component (test ${source} NAME_WE)		# filename without extension
    add_executable (${test} ${source})
    add_test (${test} ${EXECUTABLE_OUTPUT_PATH}/${test})
endforeach (source)


