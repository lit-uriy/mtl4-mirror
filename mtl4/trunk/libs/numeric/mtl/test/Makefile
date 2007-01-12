CXXFLAGS+=  -I${MTL_BOOST_ROOT} -I${BOOST_ROOT} -O3 -DNDEBUG -ffast-math  -funsafe-math-optimizations -ffinite-math-only 

CXXFLAGS+= -Wno-long-double

# on Big Red with IBM Compiler
#CXX= xlC

EXECUTABLES= complexity compressed2D_inserter compressed2D dense2D unroll unroll_matrix_mult

it: ${EXECUTABLES}

clean:
	rm -f ${EXECUTABLES} *~ *.o