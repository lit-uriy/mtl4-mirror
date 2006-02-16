INCLUDE_DIRS = -I${HOME}/mtl4/branches/morton -I${HOME}/mtl4/trunk -I{HOME}/software/boost_1_33_1

CXX_FLAGS = ${INCLUDE_DIRS}

all: morton_dense


morton_dense:	morton_dense.cpp
	g++ morton_dense.cpp -o morton_dense ${CXX_FLAGS}

clean:
	rm morton_dense
