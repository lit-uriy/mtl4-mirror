INCLUDE_DIRS = -I${MTL_MORTON_BOOST_ROOT} -I${MTL_BOOST_ROOT} -I${BOOST_ROOT}

CXX_FLAGS = ${INCLUDE_DIRS}

all: morton_dense


morton_dense:	morton_dense.cpp
	g++ morton_dense.cpp -o morton_dense ${CXX_FLAGS}

clean:
	rm morton_dense



