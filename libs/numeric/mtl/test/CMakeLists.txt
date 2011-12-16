project (mtl4_tests)

# collect the source files
file (GLOB SOURCES *_test.cpp)
compile_all("true" ${SOURCES})
compile_all("true" "vpt_test_1.cpp vpt_test_2.cpp")
