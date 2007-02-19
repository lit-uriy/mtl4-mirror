INCLUDE_DIRS = -I${MTL_MORTON_BOOST_ROOT} -I${MTL_BOOST_ROOT} -I${BOOST_ROOT}

CXXFLAGS = ${INCLUDE_DIRS} 

CXXFLAGS += -Wno-long-double 

# Debugging
CXXFLAGS += -g

# light optimization
#CXXFLAGS += -DNDEBUG -O2

# g++ optimization
# CXXFLAGS += -DNDEBUG -O3 -ffast-math 

# Additional G5 optimization on g++
# CXXFLAGS += -fast

# Intel optimization
#CXXFLAGS += -DNDEBUG -O3 

# To use Michaels assembly code (is not used when not compiled with g++)
#CXXFLAGS += -DMTL_USE_OPTERON_OPTIMIZATION


# CXX = icc


clean:
	rm morton_dense matrix_product *~



