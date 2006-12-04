INCLUDE_DIRS = -I${MTL_MORTON_BOOST_ROOT} -I${MTL_BOOST_ROOT} -I${BOOST_ROOT}

CXXFLAGS = ${INCLUDE_DIRS} -Wno-long-double # -g
CXXFLAGS += -DNDEBUG -O3 -fast  -ffast-math  -funsafe-math-optimizations -ffinite-math-only -fast

all: morton_dense matrix_product


#morton_dense:	morton_dense.cpp
#	g++ morton_dense.cpp -o morton_dense ${CXX_FLAGS}

clean:
	rm morton_dense matrix_product *~



