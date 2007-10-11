INCLUDE_DIRS = -I${MTL_BOOST_ROOT} -I${BOOST_ROOT}

CXXFLAGS = ${INCLUDE_DIRS} 

#CXXFLAGS += -Wno-long-double 

# Debugging
#CXXFLAGS += -g

# g++ optimization
#CXXFLAGS += -DNDEBUG -O3 -ffast-math 

# Additional G5 optimization on g++
# CXXFLAGS += -fast

# Intel Compiler
CXX= icc

# Intel optimization
#CXXFLAGS += -DNDEBUG -O3 

# on Big Red with IBM Compiler
#CXX= xlC

EXECUTABLES= complexity compressed2D_inserter compressed2D dense2D unroll unroll_matrix_mult
EXECS=$(shell ls *.cpp | sed -e 's/\.cpp//')

it: ${EXECS}

clean:
	rm -f ${EXECUTABLES} *~ *.o
