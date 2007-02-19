INCLUDE_DIRS = -I${MTL_MORTON_BOOST_ROOT} -I${MTL_BOOST_ROOT} -I${BOOST_ROOT}

CXXFLAGS = ${INCLUDE_DIRS} -g -Wno-long-double 

# on Big Red with IBM Compiler
#CXX= xlC


EXECUTABLES = morton_dense sub_matrix matrix_product_test

#all: ${EXECUTABLES}


#morton_dense:	morton_dense.cpp
#	g++ morton_dense.cpp -o morton_dense ${CXX_FLAGS}

clean:
	rm morton_dense
