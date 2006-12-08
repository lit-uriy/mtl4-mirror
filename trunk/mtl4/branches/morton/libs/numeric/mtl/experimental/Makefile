INCLUDE_DIRS = -I${MTL_MORTON_BOOST_ROOT} -I${MTL_BOOST_ROOT} -I${BOOST_ROOT}

CXXFLAGS = ${INCLUDE_DIRS} 

# CXXFLAGS += -Wno-long-double 

# Debugging
# CXXFLAGS += -g

# g++ optimization
CXXFLAGS += -DNDEBUG -O3 -ffast-math  -funsafe-math-optimizations -ffinite-math-only 

# Additional G5 optimization on g++
# CXXFLAGS += -fast

# Intel optimization
# CXXFLAGS += -DNDEBUG -O3 

# To use Michaels assembly code (is not used when not compiled with g++)
CXXFLAGS += -DMTL_USE_OPTERON_OPTIMIZATION


# CXX = icc

all: morton_dense matrix_product


#morton_dense:	morton_dense.cpp
#	g++ morton_dense.cpp -o morton_dense ${CXX_FLAGS}

clean:
	rm morton_dense matrix_product *~



