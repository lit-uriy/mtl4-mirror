INCLUDE_DIRS = -I${MTL_MORTON_BOOST_ROOT} -I${MTL_BOOST_ROOT} -I${BOOST_ROOT}

CXXFLAGS = ${INCLUDE_DIRS} 

# CXXFLAGS += -Wno-long-double 

# Debugging
#CXXFLAGS += -g

# g++ optimization
CXXFLAGS += -DNDEBUG -O3 -ffast-math 

# Additional G5 optimization on g++
# CXXFLAGS += -fast

# Intel optimization
# CXXFLAGS += -DNDEBUG -O3 

# To use Michaels assembly code (is not used when not compiled with g++)
#CXXFLAGS += -DMTL_USE_OPTERON_OPTIMIZATION


# CXX = icc


specialize_mult_dispatching_timing: specialize_mult_dispatching_timing.cpp ../src/mult_add_base_case_32_shark_2_impl.cpp
	${CXX} ${CXXFLAGS} $? -o $@


#morton_dense:	morton_dense.cpp
#	g++ morton_dense.cpp -o morton_dense ${CXX_FLAGS}

clean:
	rm morton_dense matrix_product *~



